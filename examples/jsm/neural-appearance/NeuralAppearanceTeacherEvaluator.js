import * as THREE from 'three';
import * as TSL from 'three/tsl';
import {
	computeAtlasDimensions,
	createAtlasTextures,
	createAtlasShaderNodes,
	createTeacherRenderTarget,
	uploadAtlasSamples
} from './NeuralAppearanceTeacherAtlas.js';
import {
	readSamplePixel,
	renderAndReadTeacherAttachments
} from './NeuralAppearanceTeacherReadback.js';
import { resolveOpacityMode } from './NeuralAppearanceFormat.js';
import {
	NeuralTeacherLightingModel,
	NeuralTeacherIBLLightingModel,
	NeuralTeacherIBLEnvironmentNode,
	createTeacherIBLQueryNodes
} from './NeuralAppearanceTeacherLightingModels.js';

const DEFAULT_BATCH_SIZE = 1024;
const DEFAULT_TILE_SIZE = 8;
const DEFAULT_UV_GRADIENT_SCALE = 1 / 1024;

// Maps a public single-purpose targetMode onto the merged MRT render group it
// shares a draw call with, plus the name/size of its attachment within that
// group's output. evaluateBatch()'s per-call contract (a flat array for the
// requested mode) is unchanged -- see _evaluateGrouped() for how this is used
// to serve several modes from one render+readback when they're requested
// back-to-back for the *same* sample batch, which is how
// NeuralAppearanceSampler.js already calls them (assignTeacherTargets then
// assignAuxiliaryTeacherTargets share one `samples` array; the five
// evaluateBatch(teacherSamples, ...) IBL calls in assignIBLTeacherTargets
// share another).
//
// IBL is split into two MRT groups instead of one merged 5-attachment group:
// 'iblProbe' (query, incoming, irradiance -- 3 half-float RGBA attachments,
// 24 bytes/sample) and 'iblIndirect' (indirectRadiance, indirectIrradiance --
// 2 attachments, 16 bytes/sample). A single 5-attachment group (40
// bytes/sample) was tried first and reverted -- it failed pipeline creation
// ("Bytes per sample(40) exceeded maximum allowed limit(32)") on real
// hardware reporting maxColorAttachmentBytesPerSample=32, a tier that exists
// despite web3dsurvey.com/webgpu showing only ~2% of *surveyed* devices below
// 64 -- evidently not rare enough to assume away. Both groups here stay under
// 32 bytes/sample with headroom, which every WebGPU device is required to
// support (the spec's minimum default limit). Do not re-merge these two
// groups without first confirming the real device-limit floor, not just the
// survey's reported distribution.
// Only 'opacity' is still ungrouped (it needs alpha-testing disabled, which
// would change how the 'direct' group's fragments discard if merged with it)
// -- see NeuralAppearanceTeacherEvaluator._createResources() for why.
const GROUP_BY_MODE = {
	brdf: { groupId: 'direct', channel: 'output', size: 3 },
	emission: { groupId: 'direct', channel: 'emission', size: 3 },
	iblQuery: { groupId: 'iblProbe', channel: 'query', size: 4 },
	iblIncoming: { groupId: 'iblProbe', channel: 'incoming', size: 3 },
	iblIrradiance: { groupId: 'iblProbe', channel: 'irradiance', size: 3 },
	iblIndirectRadiance: { groupId: 'iblIndirect', channel: 'indirectRadiance', size: 3 },
	iblIndirectIrradiance: { groupId: 'iblIndirect', channel: 'indirectIrradiance', size: 3 }
};

class NeuralAppearanceTeacherEvaluator {

	constructor( material, renderer, options = {} ) {

		this.material = material;
		this.renderer = renderer;
		this.batchSize = options.teacherBatchSize || options.batchSize || DEFAULT_BATCH_SIZE;
		this.tileSize = options.teacherTileSize || DEFAULT_TILE_SIZE;
		this.uvGradientScale = options.uvGradientScale || DEFAULT_UV_GRADIENT_SCALE;
		this.sourceResolution = options.sourceResolution || options.resolution || 1;
		this.supportsEmission = material.emissiveNode !== null && material.emissiveNode !== undefined;
		this.supportsOpacity = material.opacityNode !== null && material.opacityNode !== undefined;
		this.alphaCutoff = Number.isFinite( material.alphaTest ) ? material.alphaTest : 0.5;

		const materialOpacityMode = ( material && material.userData && material.userData.opacityMode ) ? material.userData.opacityMode : ( material ? material._opacityMode : undefined );
		const heuristicOpacityMode = this.supportsOpacity && material && material.transparent === true && ( ! Number.isFinite( material.alphaTest ) || material.alphaTest <= 0 ) ?
			'blend' : 'mask';
		this.opacityMode = resolveOpacityMode( options.opacityMode, materialOpacityMode, heuristicOpacityMode );

		this.environment = options.environment || null;
		this.supportsIBL = this.environment !== null && this.environment !== undefined;

		// Per-target-mode resource bundles ({ scene, camera, geometry, material,
		// mesh, light, target }), built lazily on first use and cached for the
		// lifetime of this evaluator -- avoids the dispose()/rebuild/shader-
		// recompile cycle that used to happen every time evaluateBatch() was
		// called with a different targetMode than the previous call.
		this._modeBundles = new Map();
		this._atlasWidth = 0;
		this._atlasHeight = 0;
		this._atlasColumns = 0;
		this._sampleTextures = null;
		this._initialized = false;

		// One-slot cache for the most recently rendered MRT group (see
		// GROUP_BY_MODE): if the next evaluateBatch() call requests a mode
		// from the *same* group for the *same* `samples` array reference, its
		// result is already in hand and no render/readback is needed at all.
		this._lastGroupId = null;
		this._lastGroupSamples = null;
		this._lastGroupResults = null;

	}

	async init() {

		if ( this._initialized ) return;

		if ( ! this.renderer || this.renderer.isWebGPURenderer !== true ) {

			throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: WebGPU renderer is required for MaterialX node-material teacher sampling.' );

		}

		if ( this.renderer.init ) await this.renderer.init();

		this._createSharedAtlasResources();
		this._initialized = true;

	}

	encodeInputs( uv = [ 0.5, 0.5 ] ) {

		return [
			uv[ 0 ], uv[ 1 ],
			0, 0, 1,
			1, 0, 0,
			0, 1, 0,
			0, 0, 1
		];

	}

	async evaluate( sample ) {

		const results = await this.evaluateBatch( [ sample ] );
		return results[ 0 ];

	}

	async evaluateBatch( samples, targetMode = 'brdf' ) {

		if ( this._initialized === false ) await this.init();
		if ( samples.length === 0 ) return [];

		const groupInfo = GROUP_BY_MODE[ targetMode ];

		return groupInfo ? this._evaluateGrouped( samples, targetMode, groupInfo ) : this._evaluateUngrouped( samples, targetMode );

	}

	// Modes that share an MRT render group (see GROUP_BY_MODE): if the
	// immediately preceding call already rendered this group for this exact
	// `samples` array, its results are reused as-is -- no render/readback.
	// Otherwise renders the whole group once (all of its channels, batched
	// the same way a single-mode pass would be) and caches every mode's
	// slice of it, not just the one requested.
	async _evaluateGrouped( samples, targetMode, groupInfo ) {

		if ( this._lastGroupId === groupInfo.groupId && this._lastGroupSamples === samples ) {

			return this._lastGroupResults[ targetMode ];

		}

		const bundle = this._getModeBundle( groupInfo.groupId );
		const modesInGroup = Object.keys( GROUP_BY_MODE ).filter( ( mode ) =>
			GROUP_BY_MODE[ mode ].groupId === groupInfo.groupId && bundle.channelNames.includes( GROUP_BY_MODE[ mode ].channel )
		);

		const results = {};
		for ( const mode of modesInGroup ) results[ mode ] = new Array( samples.length );

		for ( let offset = 0; offset < samples.length; offset += this.batchSize ) {

			const batch = samples.slice( offset, offset + this.batchSize );
			this._uploadSamples( batch );

			const pixelsByChannel = await this._renderAndRead( bundle );

			for ( let i = 0; i < batch.length; i ++ ) {

				for ( const mode of modesInGroup ) {

					const { channel, size } = GROUP_BY_MODE[ mode ];
					const pixel = this._readSamplePixel( pixelsByChannel[ channel ], i );
					results[ mode ][ offset + i ] = pixel.slice( 0, size );

				}

			}

		}

		this._lastGroupId = groupInfo.groupId;
		this._lastGroupSamples = samples;
		this._lastGroupResults = results;

		return results[ targetMode ];

	}

	// Modes with no merge group -- each gets its own single-channel pass, as
	// before Phase 2 (Phase 1's per-mode bundle cache still applies).
	async _evaluateUngrouped( samples, targetMode ) {

		const bundle = this._getModeBundle( targetMode );
		const channel = bundle.channelNames[ 0 ];
		const targets = new Array( samples.length );

		for ( let offset = 0; offset < samples.length; offset += this.batchSize ) {

			const batch = samples.slice( offset, offset + this.batchSize );
			this._uploadSamples( batch );

			const pixelsByChannel = await this._renderAndRead( bundle );

			for ( let i = 0; i < batch.length; i ++ ) {

				const pixel = this._readSamplePixel( pixelsByChannel[ channel ], i );
				targets[ offset + i ] = pixel.slice( 0, 3 );

			}

		}

		return targets;

	}

	dispose() {

		for ( const bundle of this._modeBundles.values() ) {

			if ( bundle.target ) bundle.target.dispose();
			if ( bundle.geometry ) bundle.geometry.dispose();
			if ( bundle.material && bundle.material !== this.material && bundle.material.dispose ) bundle.material.dispose();

		}

		this._modeBundles.clear();

		if ( this._sampleTextures ) {

			for ( const texture of Object.values( this._sampleTextures ) ) {

				texture.dispose();

			}

		}

		this._sampleTextures = null;
		this._initialized = false;
		this._lastGroupId = null;
		this._lastGroupSamples = null;
		this._lastGroupResults = null;

	}

	// Builds (once) the shared atlas input-sample textures and the shader
	// nodes that read from them. These encode uv/frame/direction only and are
	// mode-agnostic -- every per-mode material bundle samples from the same
	// atlas textures, so they're created once here rather than per mode.
	_createSharedAtlasResources() {

		const capacity = Math.max( 1, this.batchSize );
		const { atlasColumns, atlasRows, atlasWidth, atlasHeight } = computeAtlasDimensions( capacity, this.tileSize );
		this._atlasColumns = atlasColumns;
		this._atlasWidth = atlasWidth;
		this._atlasHeight = atlasHeight;

		this._sampleTextures = createAtlasTextures( atlasColumns, atlasRows );
		this._atlasNodes = createAtlasShaderNodes( this._sampleTextures, atlasColumns, atlasRows, this.tileSize );

	}

	// Returns the cached scene/material/render-target bundle for targetMode,
	// building and caching it on first use. Never disposed/rebuilt afterwards
	// -- this is what removes the per-iteration shader-recompile + resource-
	// churn cycle that used to happen on every targetMode switch.
	_getModeBundle( targetMode ) {

		let bundle = this._modeBundles.get( targetMode );
		if ( bundle ) return bundle;

		bundle = this._createResources( targetMode );
		this._modeBundles.set( targetMode, bundle );
		return bundle;

	}

	// `key` is either a merge-group id from GROUP_BY_MODE ('direct',
	// 'iblProbe') or, for modes with no merge group, the targetMode itself
	// ('opacity', 'iblIndirectRadiance', 'iblIndirectIrradiance'). Either way
	// it doubles as the cache key in _modeBundles / _getModeBundle().
	_createResources( key ) {

		const nodes = this._atlasNodes;

		const sampleMaterial = this.material.clone ? this.material.clone() : this.material;
		sampleMaterial.contextNode = TSL.replaceUV( nodes.materialUv, TSL.overrideNodes( [
			[ TSL.normalView, nodes.normal ],
			[ TSL.tangentView, nodes.tangent ],
			[ TSL.bitangentView, nodes.bitangent ],
			[ TSL.positionViewDirection, nodes.wo ]
		] ) );

		let light = null;
		let channelNames;
		let usesEnvironment = false;

		if ( key === 'direct' ) {

			// Merges the direct-lit BRDF color ('brdf') with the emission
			// readout ('emission', when supported) into a single MRT draw --
			// both leave alpha-testing at the teacher material's default, so
			// they discard identically. Opacity deliberately stays its own
			// pass below: it needs alpha-testing disabled to read a value
			// even where this draw would discard the fragment as masked-out.
			light = this._createLight();
			sampleMaterial.lightsNode = TSL.lights( [ light ] ).context( {
				lightingModel: new NeuralTeacherLightingModel( sampleMaterial, nodes.wi )
			} );

			channelNames = [ 'output' ];

			if ( this.supportsEmission ) {

				// Only reach for mrtNode when there's more than one attachment to
				// name -- MRTNode.setup() matches outputs to render-target
				// textures *by name* (see the naming note near the RenderTarget
				// creation below), which the plain single-attachment path
				// doesn't set up (and doesn't need to: it just lets the
				// material's default output pipeline write straight to
				// attachment 0, exactly like the un-merged 'brdf' pass used to).
				const emission = sampleMaterial.emissiveNode || TSL.vec3( 0 );
				sampleMaterial.mrtNode = TSL.mrt( { output: TSL.output, emission: TSL.vec4( emission, 1 ) } );
				channelNames.push( 'emission' );

			}

		} else if ( key === 'opacity' ) {

			const opacity = sampleMaterial.opacityNode || TSL.float( 1 );
			sampleMaterial.lights = false;
			sampleMaterial.lightsNode = TSL.lights( [] );
			sampleMaterial.alphaTest = 0;
			sampleMaterial.alphaTestNode = null;
			sampleMaterial.outputNode = TSL.vec4( opacity, opacity, opacity, 1 );
			channelNames = [ 'opacity' ];

		} else if ( key === 'iblProbe' ) {

			if ( this.environment === null ) {

				throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: An environment texture is required for IBL probe sampling.' );

			}

			// Merges the three IBL-query-style readouts ('iblQuery',
			// 'iblIncoming', 'iblIrradiance') into one MRT draw: all three
			// leave lights=false and don't touch alpha-testing, so they
			// discard identically, and 'iblIncoming' already depends on
			// 'iblQuery's direction/roughness -- computed once and shared
			// here rather than recomputed per pass. Kept separate from the
			// 'iblIndirect' group below to stay under the 32-bytes/sample
			// MRT attachment budget some real WebGPU devices enforce -- see
			// GROUP_BY_MODE's header comment.
			const query = createTeacherIBLQueryNodes( sampleMaterial );
			sampleMaterial.lights = false;
			sampleMaterial.lightsNode = TSL.lights( [] );

			const envNode = TSL.pmremTexture( this.environment );
			const incoming = envNode.context( {
				getUV: () => query.radianceDir.transformDirection( TSL.cameraWorldMatrix ),
				getTextureLevel: () => query.roughness
			} );
			// Incoming PMREM irradiance (N, mip 1). Training input only; outgoing
			// IBL is sampled with iblIndirect / PhysicalLightingModel.
			const irradiance = envNode.context( {
				getUV: () => TSL.normalWorld,
				getTextureLevel: () => TSL.float( 1 )
			} );

			sampleMaterial.mrtNode = TSL.mrt( {
				query: TSL.vec4( query.radianceDir, query.roughness ),
				incoming: TSL.vec4( incoming, 1 ),
				irradiance: TSL.vec4( irradiance, 1 )
			} );

			channelNames = [ 'query', 'incoming', 'irradiance' ];
			usesEnvironment = true;

		} else if ( key === 'iblIndirect' ) {

			if ( this.environment === null ) {

				throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: An environment texture is required for IBL teacher sampling.' );

			}

			// Merges the two BRDF-weighted indirect readouts
			// ('iblIndirectRadiance', 'iblIndirectIrradiance') into one MRT
			// draw, reading both out of a single un-isolated ('full')
			// lighting-model pass via the `capture` param instead of the two
			// differently-isolated passes this used to be (isolate='radiance'
			// / isolate='irradiance', each its own render). This is not
			// bit-identical to that old two-pass isolate hack: the old
			// 'irradiance'-isolated pass left a small multi-scattering
			// cross-term (which depends on *both* radiance and irradiance)
			// leaking into what was meant to be a pure-irradiance channel;
			// this merged pass instead reads PhysicalLightingModel's real
			// indirectDiffuse/indirectSpecular split directly, which is the
			// physically-correct split with no leak. This changes what these
			// two training targets numerically represent by a small amount --
			// compare loss curves against a pre-change baseline before relying
			// on this for anything beyond the intended speedup.
			const indirectDiffuseVar = TSL.vec3( 0 ).toVar( 'neuralTeacherIndirectDiffuse' );
			const indirectSpecularVar = TSL.vec3( 0 ).toVar( 'neuralTeacherIndirectSpecular' );

			sampleMaterial.lightsNode = TSL.lights( [] ).context( {
				lightingModel: new NeuralTeacherIBLLightingModel( sampleMaterial, 'full', {
					indirectDiffuse: indirectDiffuseVar,
					indirectSpecular: indirectSpecularVar
				} )
			} );
			sampleMaterial.setupEnvironment = function ( builder ) {

				let envNode = THREE.NodeMaterial.prototype.setupEnvironment.call( this, builder );
				if ( envNode === null && builder.environmentNode ) envNode = builder.environmentNode;
				return envNode ? new NeuralTeacherIBLEnvironmentNode( envNode, 'full' ) : null;

			};

			sampleMaterial.mrtNode = TSL.mrt( {
				indirectRadiance: TSL.vec4( indirectSpecularVar, 1 ),
				indirectIrradiance: TSL.vec4( indirectDiffuseVar, 1 )
			} );

			channelNames = [ 'indirectRadiance', 'indirectIrradiance' ];
			usesEnvironment = true;

		} else {

			throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: Unsupported target mode "${ key }".` );

		}

		// Every bundle clones the same base `this.material`, and several
		// bundles' node graphs are structurally close to each other (e.g.
		// 'iblProbe' vs 'iblIndirect' both build off the same MaterialX
		// graph with only lightsNode/mrtNode differing). Without an explicit,
		// per-`key` cache-key suffix here, two bundles' pipelines can hash to
		// the same program cache entry and get cross-used against each
		// other's render target -- which surfaces as a WebGPU
		// GPUValidationError ("color and depth targets from pass do not
		// match pipeline") since the reused pipeline was compiled for a
		// different attachment count/format than the target actually bound
		// for that draw. `key` (the mode-bundle cache key -- see
		// _getModeBundle()) is guaranteed unique per bundle, so suffixing
		// with it guarantees each bundle gets its own pipeline. `key` alone
		// is not enough across separate evaluator instances though: two
		// runs' 'direct' bundles share the same `key` but can differ in
		// attachment count (e.g. supportsEmission changes whether
		// 'emission' is included), so the suffix must also encode the
		// resolved channel/attachment signature.
		sampleMaterial.customProgramCacheKey = function () {

			return THREE.NodeMaterial.prototype.customProgramCacheKey.call( this ) + '|neuralTeacherMode:' + key + ':' + channelNames.join( ',' );

		};

		sampleMaterial.toneMapped = false;
		sampleMaterial.needsUpdate = true;

		const scene = new THREE.Scene();
		const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
		camera.position.set( 0, 0, 2 );
		const geometry = new THREE.PlaneGeometry( 2, 2 );
		const mesh = new THREE.Mesh( geometry, sampleMaterial );
		scene.add( mesh );
		if ( light ) scene.add( light );
		if ( usesEnvironment ) scene.environment = this.environment;

		// MRTNode matches each mrtNode output by attachment *name* (see
		// THREE.MRTNode#setup), so multi-channel bundles need their render
		// target's textures named to match; single-channel bundles just use
		// outputNode directly and don't need MRT/naming at all.
		const target = createTeacherRenderTarget( this._atlasWidth, this._atlasHeight, channelNames.length > 1 ? channelNames : null );

		return { scene, camera, geometry, material: sampleMaterial, mesh, light, target, channelNames };

	}

	_createLight() {

		const light = new THREE.DirectionalLight( 0xffffff, 1 );
		light.position.set( 0, 0, 1 );
		return light;

	}

	_uploadSamples( samples ) {

		uploadAtlasSamples( this._sampleTextures, samples, this._atlasColumns, this._atlasHeight, this.tileSize, this.uvGradientScale );

	}

	// Renders bundle.scene once and reads back every one of its named
	// attachments from that single draw, returned as { [channelName]: pixels }.
	async _renderAndRead( bundle ) {

		const textureIndices = bundle.channelNames.map( ( _name, i ) => i );
		const attachments = await renderAndReadTeacherAttachments( this.renderer, bundle.scene, bundle.camera, bundle.target, this._atlasWidth, this._atlasHeight, textureIndices );

		const pixelsByChannel = {};
		bundle.channelNames.forEach( ( name, i ) => {

			pixelsByChannel[ name ] = attachments[ i ];

		} );
		return pixelsByChannel;

	}

	_readSamplePixel( pixels, sampleIndex ) {

		return readSamplePixel( pixels, sampleIndex, this._atlasColumns, this._atlasWidth, this.tileSize );

	}

}

function createGpuMaterialTeacher( material, renderer, options ) {

	if ( material === null || material === undefined || material.isMeshPhysicalNodeMaterial !== true ) {

		const type = material && material.type ? material.type : typeof material;
		throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: A supported MeshPhysicalNodeMaterial teacher is required; received ${ type }.` );

	}

	return new NeuralAppearanceTeacherEvaluator( material, renderer, options );

}

export {
	NeuralAppearanceTeacherEvaluator,
	createGpuMaterialTeacher
};

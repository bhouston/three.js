import * as THREE from 'three';
import * as TSL from 'three/tsl';
import { EnvironmentNode, PhysicalLightingModel } from 'three/webgpu';
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
// assignAuxiliaryTeacherTargets share one `samples` array; the three
// evaluateBatch(teacherSamples, ...) IBL-probe calls in
// assignIBLTeacherTargets share another). Modes not listed here (opacity,
// iblIndirectRadiance, iblIndirectIrradiance) keep their own single-purpose
// pass -- see NeuralAppearanceTeacherEvaluator._createResources() for why.
const GROUP_BY_MODE = {
	brdf: { groupId: 'direct', channel: 'output', size: 3 },
	emission: { groupId: 'direct', channel: 'emission', size: 3 },
	iblQuery: { groupId: 'iblProbe', channel: 'query', size: 4 },
	iblIncoming: { groupId: 'iblProbe', channel: 'incoming', size: 3 },
	iblIrradiance: { groupId: 'iblProbe', channel: 'irradiance', size: 3 }
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
		if ( options.opacityMode === 'mask' || options.opacityMode === 'blend' ) {

			this.opacityMode = options.opacityMode;

		} else if ( materialOpacityMode === 'mask' || materialOpacityMode === 'blend' ) {

			this.opacityMode = materialOpacityMode;

		} else if ( this.supportsOpacity && material && material.transparent === true && ( ! Number.isFinite( material.alphaTest ) || material.alphaTest <= 0 ) ) {

			this.opacityMode = 'blend';

		} else {

			this.opacityMode = 'mask';

		}

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
			// here rather than recomputed per pass.
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

		} else if ( key === 'iblIndirectRadiance' || key === 'iblIndirectIrradiance' ) {

			if ( this.environment === null ) {

				throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: An environment texture is required for IBL teacher sampling.' );

			}

			const isolate = key === 'iblIndirectRadiance' ? 'radiance' : 'irradiance';

			sampleMaterial.lightsNode = TSL.lights( [] ).context( {
				lightingModel: new NeuralTeacherIBLLightingModel( sampleMaterial, isolate )
			} );
			sampleMaterial.setupEnvironment = function ( builder ) {

				let envNode = THREE.NodeMaterial.prototype.setupEnvironment.call( this, builder );
				if ( envNode === null && builder.environmentNode ) envNode = builder.environmentNode;
				return envNode ? new NeuralTeacherIBLEnvironmentNode( envNode, isolate ) : null;

			};

			sampleMaterial.customProgramCacheKey = function () {

				return THREE.NodeMaterial.prototype.customProgramCacheKey.call( this ) + '|teacherIblIsolate:' + isolate;

			};

			channelNames = [ key ];
			usesEnvironment = true;

		} else {

			throw new Error( `THREE.NeuralAppearanceTeacherEvaluator: Unsupported target mode "${ key }".` );

		}

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

class NeuralTeacherLightingModel extends PhysicalLightingModel {

	constructor( material, lightDirectionNode ) {

		if ( THREE.PhysicalLightingModel === undefined ) {

			throw new Error( 'THREE.NeuralAppearanceTeacherEvaluator: PhysicalLightingModel is required for directional GPU teacher sampling. Import three/webgpu for training.' );

		}

		super( material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useTransmission, material.useDispersion, material.useRetroreflection );
		this.lightDirectionNode = lightDirectionNode;

	}

	direct( { reflectedLight }, builder ) {

		super.direct( {
			lightDirection: this.lightDirectionNode,
			lightColor: TSL.vec3( 1 ),
			reflectedLight
		}, builder );

	}

	indirect( /*builder*/ ) {}

}

class NeuralTeacherIBLLightingModel extends PhysicalLightingModel {

	constructor( material, isolate = 'full' ) {

		super( material.useClearcoat, material.useSheen, material.useIridescence, material.useAnisotropy, material.useTransmission, material.useDispersion, material.useRetroreflection );

		this.isolate = isolate;

	}

	direct( /*input, builder*/ ) {}

}

class NeuralTeacherIBLEnvironmentNode extends EnvironmentNode {

	static get type() {

		return 'NeuralTeacherIBLEnvironmentNode';

	}

	constructor( envNode = null, isolate = 'full' ) {

		super( envNode );
		this.isolate = isolate;

	}

	customCacheKey() {

		const isolateKey = this.isolate === 'radiance' ? 1 : this.isolate === 'irradiance' ? 2 : 0;
		return super.customCacheKey() + isolateKey;

	}

	setup( builder ) {

		const { material } = builder;
		let envNode = this.envNode;

		if ( envNode.isTextureNode || envNode.isMaterialReferenceNode ) {

			const value = ( envNode.isTextureNode ) ? envNode.value : material[ envNode.property ];
			const cache = this._getPMREMNodeCache( builder.renderer );
			let cacheEnvNode = cache.get( value );

			if ( cacheEnvNode === undefined ) {

				cacheEnvNode = TSL.pmremTexture( value );
				cache.set( value, cacheEnvNode );

			}

			envNode = cacheEnvNode;

		}

		const useAnisotropy = material.useAnisotropy === true || material.anisotropy > 0;
		const radianceNormalView = useAnisotropy ? TSL.bentNormalView : TSL.normalView;
		const radiance = envNode.context( createRadianceContext( TSL.roughness, radianceNormalView ) ).mul( TSL.materialEnvIntensity );
		const irradiance = envNode.context( createIrradianceContext( TSL.normalWorld ) ).mul( Math.PI ).mul( TSL.materialEnvIntensity );

		if ( this.isolate !== 'irradiance' ) {

			builder.context.radiance.addAssign( radiance.isolate() );

		}

		if ( this.isolate !== 'radiance' ) {

			builder.context.iblIrradiance.addAssign( irradiance.isolate() );

		}

		const clearcoatRadiance = builder.context.lightingModel.clearcoatRadiance;

		if ( clearcoatRadiance && this.isolate !== 'irradiance' ) {

			clearcoatRadiance.addAssign(
				envNode.context( createRadianceContext( TSL.clearcoatRoughness, TSL.clearcoatNormalView ) ).mul( TSL.materialEnvIntensity ).isolate()
			);

		}

	}

}

function createRadianceContext( roughnessNode, normalViewNode ) {

	let reflectVec = null;

	return {
		getUV: () => {

			if ( reflectVec === null ) {

				reflectVec = TSL.positionViewDirection.negate().reflect( normalViewNode );
				reflectVec = TSL.pow4( roughnessNode ).mix( reflectVec, normalViewNode ).normalize();
				reflectVec = reflectVec.transformDirection( TSL.cameraWorldMatrix );

			}

			return reflectVec;

		},
		getTextureLevel: () => roughnessNode
	};

}

function createIrradianceContext( normalWorldNode ) {

	return {
		getUV: () => normalWorldNode,
		getTextureLevel: () => TSL.float( 1 )
	};

}

function createTeacherIBLQueryNodes( material ) {

	const shadingNormal = TSL.normalView.normalize();
	const viewDir = TSL.positionViewDirection.normalize();
	const roughnessSource = material.roughnessNode !== undefined && material.roughnessNode !== null ?
		TSL.float( material.roughnessNode ) :
		TSL.materialRoughness;
	const roughness = TSL.getRoughness( { roughness: roughnessSource } );
	const reflectDir = viewDir.negate().reflect( shadingNormal );
	const roughness4 = roughness.mul( roughness ).mul( roughness ).mul( roughness );
	const radianceDir = roughness4.mix( reflectDir, shadingNormal ).normalize();

	return { roughness, radianceDir };

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
	NeuralTeacherIBLLightingModel,
	NeuralTeacherIBLEnvironmentNode,
	createGpuMaterialTeacher
};

import * as THREE from 'three/webgpu';
import { Fn, abs, cameraProjectionMatrixInverse, cameraWorldMatrix, context, diffuseColor, float, floor, getViewPosition, materialAO, materialEmissive, materialMetalness, max, min, mrt, normalView, output, overrideNodes, passTexture, positionGeometry, positionView, positionViewDirection, positionWorld, roughness, shadow, texture, uniform, uv, vec2, vec3, vec4, velocity } from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { HistoryState, plan } from './contracts.js';

// Signed octahedral normals. sign(0) must be +1 at the south pole.
const signNotZero = ( v ) => v.greaterThanEqual( 0 ).select( 1, - 1 );
const octEncode = Fn( ( [ normal ] ) => {

	const n = normal.div( abs( normal.x ).add( abs( normal.y ), abs( normal.z ) ) );
	return n.z.lessThan( 0 ).select( n.yx.abs().oneMinus().mul( vec2( signNotZero( n.x ), signNotZero( n.y ) ) ), n.xy );

} );
const octDecode = Fn( ( [ encoded ] ) => {

	const n = vec3( encoded, float( 1 ).sub( abs( encoded.x ), abs( encoded.y ) ) ).toVar();
	const t = max( n.z.negate(), 0 );
	n.xy.addAssign( vec2( n.x.greaterThanEqual( 0 ).select( t.negate(), t ), n.y.greaterThanEqual( 0 ).select( t.negate(), t ) ) );
	return n.normalize();

} );

// Neutral screen-space radiance replaces environment radiance only where a hit has coverage.
// The physical model applies its BRDF once, after this choice; diffuse IBL is unchanged.
class ResolveEnvironment extends THREE.EnvironmentNode {

	constructor( environment, radiance, metadata ) {

		super( texture( environment ) );
		this.signal = radiance;
		this.metadata = metadata;

	}

	setup( builder ) {

		super.setup( builder );
		const validDistance = this.metadata.g.greaterThan( 0 ).and( this.metadata.g.lessThanEqual( 65504 ) );
		const coverage = validDistance.select( this.metadata.r.clamp(), 0 );
		builder.context.radiance.assign( builder.context.radiance.mul( coverage.oneMinus() ).add( this.signal.rgb.mul( coverage ) ) );

	}

}

class ResolveMaterial extends THREE.MeshStandardNodeMaterial {

	setupEnvironment() {

		return this.deferredEnvironment;

	}

}

// The node owns explicit ordered passes. Constants cannot prune a materialized stage.
// Raw texture inputs inside stage shaders have no implicit scene-render dependencies.
export class GBufferPipeline extends THREE.Node {

	static get type() {

		return 'GBufferThroughputNode';

	}

	constructor( renderer, scene, camera, options, lights ) {

		super( 'vec4' );
		Object.assign( this, { renderer, scene, camera, options, lights } );
		this.updateBeforeType = THREE.NodeUpdateType.FRAME;
		this.enabled = plan( options );
		this.targets = [];
		this.materials = [];
		this.steps = [];
		this.inspection = [];
		this.history = new HistoryState();
		this.frame = 0;
		this.size = new THREE.Vector2();
		this.invSize = uniform( new THREE.Vector2() );
		this.unlit = new THREE.Lighting();
		this.unlit.enabled = false;
		this.prepassMaterials = new Map();
		this.resolvedLayer = 2;

		this.depth = new THREE.DepthTexture( 1, 1, THREE.FloatType );
		this.depth.name = 'surface.depth32float';
		this.opaque = this.target( 'opaque.linearHDR', this.enabled.deferred ? {} : { depthBuffer: true, depthTexture: this.depth } );
		this.opaque.texture.name = 'output';
		this.historyTargets = this.enabled.deferred ? [ this.target( 'source.history0' ), this.target( 'source.history1' ) ] : [];
		this.historyDepth = this.historyTargets.map( ( target, index ) => {

			const depth = new THREE.DepthTexture( 1, 1, THREE.FloatType );
			depth.name = 'source.depthHistory' + index;
			return depth;

		} );
		this.renderTarget = this.enabled.deferred ? this.target( 'beauty.linearHDR', { depthBuffer: true, depthTexture: this.depth } ) : this.opaque;
		this.renderTarget.texture.name = 'output';
		this.geometryObjects = [];
		this.hasFallback = false;
		this.geometryHistory = [ null, null ];
		scene.traverse( object => {

			if ( ! object.isMesh ) return;
			this.geometryObjects.push( object );
			if ( object.layers.isEnabled( 1 ) && ( Array.isArray( object.material ) ? object.material : [ object.material ] ).some( material => ! material.transparent && ! ( material.transmission > 0 ) ) ) this.hasFallback = true;

		} );
		this.beauty = passTexture( this, this.renderTarget.texture );
		this.depthNode = texture( this.depth );
		this.velocityNode = null;
		this.inspector( 'Opaque linear HDR', texture( this.opaque.texture ) );
		if ( this.enabled.deferred ) {

			this.previousSourceNode = texture( this.historyTargets[ 0 ].texture );
			this.previousDepthNode = texture( this.historyDepth[ 0 ] );
			this.inspector( 'Previous opaque source (pre-transparent, pre-AA)', this.previousSourceNode );
			this.inspector( 'Previous source matching depth', this.previousDepthNode );

		}

		this.inspector( 'Current depth', this.depthNode );
		this.inspector( 'Beauty before AA', texture( this.renderTarget.texture ) );

		if ( this.enabled.deferred ) this.createDeferred();
		else this.createForward();

		this.post = new THREE.RenderPipeline( renderer );
		let finalNode = this.beauty;
		if ( options.traa ) {

			this.temporal = traa( this.beauty, this.depthNode, this.velocityNode, camera );
			finalNode = this.temporal.getTextureNode();

		}

		if ( options.sharpen ) {

			const input = finalNode;
			const center = input.sample( uv() );
			const neighbours = input.sample( uv().add( vec2( this.invSize.x, 0 ) ) ).rgb
				.add( input.sample( uv().sub( vec2( this.invSize.x, 0 ) ) ).rgb,
					input.sample( uv().add( vec2( 0, this.invSize.y ) ) ).rgb,
					input.sample( uv().sub( vec2( 0, this.invSize.y ) ) ).rgb );
			finalNode = vec4( center.rgb.mul( 1.4 ).sub( neighbours.mul( 0.1 ) ).max( 0 ), 1 );

		}

		this.post.outputNode = finalNode.toInspector( 'Final beauty (linear, before output transform)' );
		this.post.outputColorTransform = options.output;
		this.resize();

	}

	target( name, options = {} ) {

		const target = new THREE.RenderTarget( 1, 1, { type: THREE.HalfFloatType, depthBuffer: false, ...options } );
		target.texture.name = name;
		this.targets.push( target );
		return target;

	}

	inspector( name, node ) {

		const inspected = node.toInspector( name );
		this.inspection.push( inspected );
		return inspected;

	}

	fullscreen( name, target, node ) {

		const material = new THREE.NodeMaterial();
		material.name = name;
		material.fragmentNode = node;
		material.depthTest = false;
		material.depthWrite = false;
		this.materials.push( material );
		const quad = new THREE.QuadMesh( material );
		quad.name = name;
		this.steps.push( { name, target, quad } );
		return texture( target.texture );

	}

	createForward() {

		this.forwardMRT = this.options.traa ? mrt( { output, velocity } ) : null;
		if ( this.options.traa ) {

			const motion = this.opaque.texture.clone();
			motion.name = 'velocity';
			motion.format = THREE.RGFormat;
			this.opaque.textures.push( motion );
			this.velocityNode = texture( motion );
			this.inspector( 'Unjittered motion (NDC)', this.velocityNode );

		}

	}

	createDeferred() {

		const packedEmission = this.renderer.backend.device.features.has( 'rg11b10ufloat-renderable' );
		this.surface = this.target( 'albedoId', { type: THREE.UnsignedByteType, depthBuffer: true, depthTexture: this.depth } );
		if ( this.enabled.prepass ) {

			this.depthOnly = new THREE.RenderTarget( 1, 1, { count: 0, depthBuffer: true, depthTexture: this.depth } );
			this.targets.push( this.depthOnly );

		}

		const specs = [
			[ 'normal', THREE.RGFormat, THREE.HalfFloatType ],
			[ 'roughMetal', THREE.RGFormat, THREE.UnsignedByteType ],
			[ 'velocity', THREE.RGFormat, THREE.HalfFloatType ],
			[ 'emissive', packedEmission ? THREE.RGBFormat : THREE.RGBAFormat, packedEmission ? THREE.UnsignedInt101111Type : THREE.HalfFloatType ]
		];
		for ( const [ name, format, type ] of specs ) {

			const attachment = this.surface.texture.clone();
			Object.assign( attachment, { name, format, type } );
			this.surface.textures.push( attachment );

		}

		this.surfaceBytesPerPixel = ( packedEmission ? 18 : 22 ) + 4;
		const limits = this.renderer.backend.device.limits;
		// WebGPU attachment cost uses 8 bytes for rg11b10ufloat (storage is 4).
		if ( limits.maxColorAttachments < 5 || limits.maxColorAttachmentBytesPerSample < 22 ) throw new Error( 'Adapter cannot support the compact surface MRT.' );
		this.surfaceMRT = mrt( {
			albedoId: vec4( diffuseColor.rgb, floor( materialAO.clamp().mul( 127 ).add( 0.5 ) ).mul( 2 ).add( 1 ).div( 255 ) ),
			normal: octEncode( normalView ),
			roughMetal: vec2( roughness, materialMetalness ),
			velocity,
			emissive: materialEmissive
		} );
		const [ albedo, normal, roughMetal, motion, emission ] = this.surface.textures.map( attachment => this.inspector( attachment.name, texture( attachment ) ) );
		this.velocityNode = motion;
		if ( this.hasFallback && this.options.traa ) {

			this.renderTarget.textures.push( this.surface.textures[ 3 ] );
			this.fallbackMRT = mrt( { output, velocity } );

		}

		const decodedNormal = octDecode( normal.rg );
		let visibility = float( 1 );
		if ( this.enabled.ao ) {

			// GTAONode requires a sampleable normal texture. Its built-in depth-normal
			// reconstruction avoids allocating an expanded normal just for this optional stage.
			this.gtao = ao( this.depthNode, null, this.camera );
			this.gtao.updateBeforeType = THREE.NodeUpdateType.NONE;
			visibility = this.gtao.getTextureNode().r;
			this.inspector( 'GTAO scalar (no bent-normal estimation)', visibility );

		}

		let indirect = vec3( 0 ), radiance = vec3( 0 ), metadata = vec2( 0, 65504 );
		if ( this.enabled.materialized ) {

			const aoTarget = this.target( 'AO visibility + surface normal fallback' );
			const aoTexture = this.fullscreen( 'AO / bent-normal contract fill', aoTarget, vec4( decodedNormal, visibility ) );
			visibility = this.inspector( 'AO + bent normal fallback', aoTexture ).a;
			const gi = this.fullscreen( 'Local SSGI neutral irradiance fill', this.target( 'SSGI.raw' ), vec4( 0, 0, 0, 1 ) );
			this.inspector( 'Local SSGI raw irradiance = 0', gi );
			const ssr = this.fullscreen( 'SSR no-hit radiance fill', this.target( 'SSR.radiance' ), vec4( 0 ) );
			this.inspector( 'SSR raw radiance = 0', ssr );
			const metaTarget = this.target( 'SSR.coverage-distance', { format: THREE.RGFormat } );
			metadata = this.inspector( 'SSR coverage / finite hit-distance', this.fullscreen( 'SSR metadata fill', metaTarget, vec4( 0, 65504, 0, 0 ) ) ).rg;
			radiance = this.inspector( 'Specular denoiser copy', this.fullscreen( 'Specular denoiser pass-through', this.target( 'SSR.filtered' ), ssr ) );
			indirect = this.inspector( 'Diffuse denoiser copy', this.fullscreen( 'Diffuse denoiser pass-through', this.target( 'SSGI.filtered' ), gi ) ).rgb;

		}

		const viewPosition = getViewPosition( uv(), this.depthNode, cameraProjectionMatrixInverse );
		const worldPosition = cameraWorldMatrix.mul( vec4( viewPosition, 1 ) ).xyz;
		const material = new ResolveMaterial();
		material.name = 'Fullscreen deferred metallic/roughness PBR';
		material.colorNode = Fn( () => {

			this.depthNode.greaterThanEqual( 1 ).discard();
			return albedo.rgb;

		} )();
		material.roughnessNode = roughMetal.r;
		material.metalnessNode = roughMetal.g;
		material.emissiveNode = emission.rgb;
		const bakedAO = floor( albedo.a.mul( 255 ).add( 0.5 ).div( 2 ) ).div( 127 );
		// min avoids multiplying the same cavity twice when baked and screen AO overlap.
		material.aoNode = min( bakedAO, visibility );
		material.receivedShadowPositionNode = worldPosition;
		material.vertexNode = vec4( positionGeometry.xy, 0, 1 );
		material.depthTest = false;
		material.depthWrite = false;
		const overrides = overrideNodes( [[ positionView, viewPosition ], [ positionViewDirection, viewPosition.negate().normalize() ], [ positionWorld, worldPosition ], [ normalView, decodedNormal ]] );
		material.contextNode = context( { ...overrides.getFlowContextData(), getGI: () => indirect } );
		material.deferredEnvironment = new ResolveEnvironment( this.scene.environment, radiance, metadata );
		this.materials.push( material );
		this.resolveMesh = new THREE.QuadMesh( material );
		this.resolveMesh.name = 'Fullscreen deferred lighting resolve';
		this.resolveMesh.layers.set( this.resolvedLayer );
		this.resolveMesh.receiveShadow = true;
		this.resolveMesh.frustumCulled = false;
		this.scene.add( this.resolveMesh );

	}

	setup( builder ) {

		// Register every buffer with the real Inspector without introducing extra draws.
		for ( const inspected of this.inspection ) inspected.build( builder );
		return texture( this.renderTarget.texture );

	}

	async compile() {

		const { renderer, scene, camera } = this;
		const state = THREE.RendererUtils.resetRendererState( renderer );
		const mask = camera.layers.mask;
		try {

			// Initialize the shared-depth OWNER first. Otherwise its first attachment
			// initialization can replace a GPU texture already sampled by a compiled
			// resolve/AA binding earlier in the frame.
			const owner = this.depth.renderTarget;
			if ( owner ) renderer.initRenderTarget( owner );
			for ( const target of this.targets ) renderer.initRenderTarget( target );
			for ( const light of this.lights ) {

				light.castShadow = this.options.shadows;
				if ( this.options.shadows && ! light.shadow.shadowNode ) {

					light.shadow.shadowNode = shadow( light );
					light.shadow.shadowNode.updateBeforeType = THREE.NodeUpdateType.NONE;

				}

			}

			if ( this.enabled.deferred ) {

				camera.layers.set( this.resolvedLayer );
				renderer.setRenderTarget( this.opaque );
				await renderer.compileAsync( scene, camera );

			} else {

				camera.layers.set( 0 );
				camera.layers.enable( 1 );
				renderer.setRenderTarget( this.opaque );
				renderer.setMRT( this.forwardMRT );
				await renderer.compileAsync( scene, camera );

			}

		} finally {

			camera.layers.mask = mask;
			THREE.RendererUtils.restoreRendererState( renderer, state );

		}

	}

	resize() {

		this.renderer.getDrawingBufferSize( this.size );
		for ( const target of this.targets ) target.setSize( this.size.x, this.size.y );
		this.depth.image.width = this.size.x;
		this.depth.image.height = this.size.y;
		this.depth.needsUpdate = true;
		for ( const depth of this.historyDepth ) {

			depth.image.width = this.size.x;
			depth.image.height = this.size.y;
			depth.needsUpdate = true;

		}

		this.invSize.value.set( 1 / this.size.x, 1 / this.size.y );
		this.history.reset( 'resize / pipeline rebuild' );

	}

	renderGeometry( target, lighting, mask, opaque, transparent, attachment = null, clear = true ) {

		const { renderer, camera } = this;
		camera.layers.mask = mask;
		renderer.lighting = lighting;
		renderer.opaque = opaque;
		renderer.transparent = transparent;
		renderer.autoClear = clear;
		renderer.setRenderTarget( target );
		renderer.setMRT( attachment );
		renderer.render( this.scene, camera );

	}

	depthPrepass() {

		const originals = [];
		this.scene.traverse( object => {

			if ( ! object.isMesh || ! object.layers.isEnabled( 0 ) ) return;
			const source = Array.isArray( object.material ) ? object.material : [ object.material ];
			const depths = source.map( material => {

				let depthMaterial = this.prepassMaterials.get( material );
				if ( ! depthMaterial ) {

					// Clone per source so coverage, side, maps, displacement and deformation agree.
					depthMaterial = material.clone();
					depthMaterial.colorWrite = false;
					this.prepassMaterials.set( material, depthMaterial );

				}

				return depthMaterial;

			} );
			originals.push( [ object, object.material ] );
			object.material = Array.isArray( object.material ) ? depths : depths[ 0 ];

		} );
		try {

			this.renderGeometry( this.depthOnly, this.unlit, 1, true, false );

		} finally {

			for ( const [ object, material ] of originals ) object.material = material;

		}

	}

	updateBefore( frame ) {

		const { renderer, scene, camera } = this;
		const state = THREE.RendererUtils.resetRendererState( renderer );
		const mask = camera.layers.mask, background = scene.background;
		const flags = { lighting: renderer.lighting, opaque: renderer.opaque, transparent: renderer.transparent, autoClearColor: renderer.autoClearColor, autoClearDepth: renderer.autoClearDepth, autoClearStencil: renderer.autoClearStencil };
		const lighting = renderer.lighting;
		this.frame ++;
		this.executed = [];
		try {

			// Shadow nodes are compiled before the first frame and share the same main camera.
			if ( this.options.shadows ) for ( const light of this.lights ) {

				light.shadow.shadowNode.updateBefore( { ...frame, renderer, scene, camera } );
				this.executed.push( 'shadow map: ' + light.name );

			}

			if ( this.enabled.deferred ) {

				scene.background = null;
				renderer.setClearColor( 0, 0 );
				if ( this.enabled.prepass ) {

					this.depthPrepass();
					this.executed.push( 'depth-only prepass (color writes disabled)' );

				}

				const depthStates = new Map();
				if ( this.enabled.prepass ) scene.traverse( object => {

					if ( ! object.isMesh || ! object.layers.isEnabled( 0 ) ) return;
					for ( const material of Array.isArray( object.material ) ? object.material : [ object.material ] ) {

						if ( depthStates.has( material ) ) continue;
						depthStates.set( material, [ material.depthFunc, material.depthWrite ] );
						material.depthFunc = THREE.EqualDepth;
						material.depthWrite = false;

					}

				} );
				renderer.autoClearDepth = ! this.enabled.prepass;
				try {

					this.renderGeometry( this.surface, this.unlit, 1, true, false, this.surfaceMRT );

				} finally {

					for ( const [ material, [ depthFunc, depthWrite ]] of depthStates ) Object.assign( material, { depthFunc, depthWrite } );

				}

				this.executed.push( 'surface MRT (5 attachments; no position)' );
				renderer.autoClearDepth = false;
				if ( this.gtao ) {

					this.gtao.updateBefore( frame );
					this.executed.push( 'GTAO scalar' );

				}

				for ( const { name, target, quad } of this.steps ) {

					renderer.setMRT( null );
					renderer.setRenderTarget( target );
					quad.render( renderer );
					this.executed.push( name );

				}

				scene.background = background;
				this.renderGeometry( this.opaque, lighting, 1 << this.resolvedLayer, true, false );
				this.executed.push( 'fullscreen deferred PBR lighting resolve' );
				// Only unsupported opaque objects, excluded from the surface traversal, use forward.
				if ( this.hasFallback ) {

					scene.background = null;
					renderer.initRenderTarget( this.renderTarget );
					renderer.copyTextureToTexture( this.opaque.texture, this.renderTarget.texture );
					this.renderGeometry( this.renderTarget, lighting, 1 << 1, true, false, this.fallbackMRT || null, false );
					renderer.copyTextureToTexture( this.renderTarget.texture, this.opaque.texture );
					this.executed.push( 'restricted forward opaque extensions + 2 HDR copies (layer 1)' );

				}

			} else {

				this.renderGeometry( this.opaque, lighting, 3, true, false, this.forwardMRT );
				this.executed.push( 'forward opaque beauty' + ( this.options.traa ? ' + motion MRT' : '' ) );

			}

			// Publish BEFORE transparency / AA / output. Read is immutable until the next frame.
			const write = this.history.write;
			this.previousMetadata = this.history.valid ? this.history.metadata[ this.history.read ] : null;
			if ( this.enabled.deferred ) {

				renderer.initRenderTarget( this.historyTargets[ write ] );
				renderer.copyTextureToTexture( this.opaque.texture, this.historyTargets[ write ].texture );
				renderer.copyTextureToTexture( this.depth, this.historyDepth[ write ] );
				this.previousSourceNode.value = this.historyTargets[ this.history.read ].texture;
				this.previousDepthNode.value = this.historyDepth[ this.history.read ];
				// Immutable CPU geometry publication accompanies the source color/depth.
				// This deliberately measures its copy cost; fake signals never consume it.
				this.geometryHistory[ write ] = this.geometryObjects.map( object => ( {
					id: object.id,
					world: new Float32Array( object.matrixWorld.elements ),
					bones: object.skeleton ? object.skeleton.boneMatrices.slice() : null,
					instances: object.isInstancedMesh ? object.instanceMatrix.array.slice() : null,
					morph: object.morphTargetInfluences ? Float32Array.from( object.morphTargetInfluences ) : null
				} ) );
				this.executed.push( 'publish opaque HDR + depth + geometry history (ping-pong)' );

			}

			this.history.publish( { frame: this.frame, projection: camera.projectionMatrix.toArray(), unjitteredProjection: ( this.temporal?._originalProjectionMatrix || camera.projectionMatrix ).toArray(), view: camera.matrixWorldInverse.toArray(), modelEpoch: this.options.model, width: this.size.x, height: this.size.y, geometrySlot: write, geometryFrame: this.frame, previousGeometryFrame: frame.frameId } );
			if ( this.enabled.deferred ) {

				renderer.initRenderTarget( this.renderTarget );
				renderer.copyTextureToTexture( this.opaque.texture, this.renderTarget.texture );
				this.executed.push( 'seed transparent framebuffer (HDR copy)' );

			}

			scene.background = null; // A Color background forces a clear even with autoClear=false.
			this.renderGeometry( this.renderTarget, lighting, 3, false, true, null, false );
			this.executed.push( 'forward transparents / transmission' );
			if ( this.options.traa ) this.executed.push( 'TRAA resolve + color/depth history copies' );
			this.executed.push( ( this.options.sharpen ? 'sharpen + ' : '' ) + ( this.options.tonemap ? 'tonemap + ' : '' ) + 'output' );

		} finally {

			scene.background = background;
			camera.layers.mask = mask;
			Object.assign( renderer, flags );
			THREE.RendererUtils.restoreRendererState( renderer, state );

		}

	}

	dispose() {

		this.scene.remove( this.resolveMesh );
		for ( const target of this.targets ) target.dispose();
		this.depth.dispose();
		for ( const depth of this.historyDepth ) depth.dispose();
		for ( const material of this.materials ) material.dispose();
		for ( const material of this.prepassMaterials.values() ) material.dispose();
		this.gtao?.dispose();
		this.temporal?.dispose();
		this.post.dispose();
		super.dispose();

	}

}

import {
	Box3,
	CubeArrayTexture,
	CubeReflectionMapping,
	HalfFloatType,
	LinearMipmapLinearFilter,
	MathUtils,
	Object3D,
	PMREMGenerator,
	RenderTarget,
	Scene,
	Vector3
} from 'three/webgpu';
import { context, indirectDiffuse, indirectSpecular } from 'three/tsl';
import { reflectionProbesDiffuse, reflectionProbesSpecular } from '../tsl/lighting/ReflectionProbesNode.js';

const _position = /*@__PURE__*/ new Vector3();
const _size = /*@__PURE__*/ new Vector3();
const _region = /*@__PURE__*/ new Box3();
const _destination = /*@__PURE__*/ new Vector3();

/**
 * A local reflection volume. Add it to a {@link ReflectionProbes} container.
 *
 * The volumes are axis-aligned boxes centered on the probe. They follow the
 * probe's world position and scale; rotation is ignored.
 *
 * @augments Object3D
 * @three_import import { ReflectionProbe } from 'three/addons/lighting/ReflectionProbes.js';
 */
class ReflectionProbe extends Object3D {

	/**
	 * Constructs a new reflection probe.
	 *
	 * @param {number} [width=1] - Width of the influence volume along X.
	 * @param {number} [height=1] - Height of the influence volume along Y.
	 * @param {number} [depth=1] - Depth of the influence volume along Z.
	 */
	constructor( width = 1, height = 1, depth = 1 ) {

		super();

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isReflectionProbe = true;

		this.type = 'ReflectionProbe';

		/**
		 * Size of the influence volume. Surfaces inside it use this probe.
		 *
		 * @type {Vector3}
		 */
		this.influenceSize = new Vector3( width, height, depth );

		/**
		 * Size of the box that reflections are projected onto, usually the room's
		 * walls. `null` uses `influenceSize`.
		 *
		 * @type {?Vector3}
		 * @default null
		 */
		this.projectionSize = null;

		/**
		 * Offset of the projection box's center from the probe, in the probe's
		 * local space. Use it when the capture point isn't at the room's center.
		 *
		 * @type {Vector3}
		 */
		this.projectionOffset = new Vector3();

		/**
		 * Distance over which the influence fades in from the volume's faces.
		 *
		 * @type {number}
		 * @default 0.5
		 */
		this.blendDistance = 0.5;

	}

	copy( source, recursive ) {

		super.copy( source, recursive );

		this.influenceSize.copy( source.influenceSize );
		this.projectionSize = source.projectionSize === null ? null : source.projectionSize.clone();
		this.projectionOffset.copy( source.projectionOffset );
		this.blendDistance = source.blendDistance;

		return this;

	}

}

/**
 * A set of {@link ReflectionProbe} children baked into one cube-array texture.
 *
 * Use `reflectionProbesSpecular( probes )` in an `indirectSpecular()` chain and
 * explicitly append an environment provider. Use `reflectionProbesDiffuse()`
 * in a diffuse chain only when no better diffuse source is available.
 * Smaller volumes take priority over larger ones; uncovered surfaces fall
 * through to the next provider. Requires the WebGPU backend.
 * The legacy `reflectionProbes()` environment node still combines both lobes
 * and the cached environment for applications without explicit lighting chains.
 *
 * ```js
 * const probes = new ReflectionProbes();
 * const probe = new ReflectionProbe( 8, 5, 8 );
 * probes.add( probe );
 * scene.add( probes );
 *
 * probes.bake( renderer, scene );
 * const environment = reflectionProbesEnvironment( probes );
 * renderer.contextNode = indirectLightingContext(
 *   indirectDiffuse( [ reflectionProbesDiffuse( probes ), environmentDiffuse( environment ) ] ),
 *   indirectSpecular( [ reflectionProbesSpecular( probes ), environmentSpecular( environment ) ] )
 * );
 * ```
 *
 * @augments Object3D
 * @three_import import { ReflectionProbes } from 'three/addons/lighting/ReflectionProbes.js';
 */
class ReflectionProbes extends Object3D {

	/**
	 * Constructs a new reflection probe set.
	 */
	constructor() {

		super();

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isReflectionProbes = true;

		this.type = 'ReflectionProbes';

		/**
		 * Whether the probes use box projection.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.boxProjection = true;

		/**
		 * The baked cube-array texture: the environment, then one cube per probe.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.texture = null;

		/**
		 * The probes in baked order, smallest volume first.
		 *
		 * @type {Array<ReflectionProbe>}
		 */
		this.bakedProbes = [];

		/**
		 * Whether a bake is in progress.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default false
		 */
		this.capturing = false;

		this._renderTarget = null;
		this._captureBounce = 0;

	}

	copy( source, recursive ) {

		super.copy( source, recursive );

		// the baked texture isn't shared, so bake the copy before use
		this.boxProjection = source.boxProjection;

		return this;

	}

	/**
	 * Captures every probe and `scene.environment` into the cube array. Hide
	 * helpers and objects that should not appear in reflections first.
	 *
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {Scene} scene - The scene to capture.
	 * @param {Object} [options] - Bake options.
	 * @param {number} [options.resolution=256] - Cube face size, rounded down to a power of two and at least 256.
	 * @param {number} [options.near=0.1] - Near plane of the capture camera.
	 * @param {number} [options.far=100] - Far plane of the capture camera.
	 * @param {boolean} [options.usePrevious=false] - Sample the last completed bake during the first pass. The previous texture remains read-only until the new bake completes.
	 * @param {number} [options.bounces=0] - Additional passes sampling the previous pass's reflections.
	 */
	bake( renderer, scene, options = {} ) {

		// cube arrays can't be emulated on WebGL 2
		if ( renderer.backend.isWebGPUBackend !== true ) throw new Error( 'ReflectionProbes: .bake() requires the WebGPU backend.' );

		const { near = 0.1, far = 100, bounces = 0, usePrevious = false } = options;

		// match the size PMREMGenerator renders at
		const resolution = Math.max( 256, MathUtils.floorPowerOfTwo( options.resolution ?? 256 ) );

		this.updateWorldMatrix( true, true );

		const probes = this.children.filter( ( child ) => child.isReflectionProbe ).sort( ( a, b ) => volume( a ) - volume( b ) );
		const maxLod = Math.log2( resolution ) - 3;

		let renderTarget = new RenderTarget( resolution, resolution, {
			depth: ( probes.length + 1 ) * 6,
			type: HalfFloatType,
			depthBuffer: false,
			generateMipmaps: false,
			minFilter: LinearMipmapLinearFilter,
			mapping: CubeReflectionMapping
		} );

		renderTarget.texture = new CubeArrayTexture().copy( renderTarget.texture );
		renderTarget.texture.name = 'ReflectionProbes';

		for ( let lod = 0; lod <= maxLod; lod ++ ) {

			renderTarget.texture.mipmaps.push( { width: resolution >> lod, height: resolution >> lod } );

		}

		renderer.initRenderTarget( renderTarget );

		const generator = new PMREMGenerator( renderer );
		const environment = new Scene();
		environment.background = scene.environment;
		environment.backgroundIntensity = scene.environmentIntensity;
		environment.backgroundRotation.copy( scene.environmentRotation );

		const previousTexture = this.texture;
		const previousProbes = this.bakedProbes;
		const hasPrevious = usePrevious && previousTexture !== null;
		const previousEnvironmentNode = scene.environmentNode;
		const previousContext = renderer.contextNode;
		const previousBackground = scene.background;
		const previousBackgroundNode = scene.backgroundNode;
		const previousBackgroundIntensity = scene.backgroundIntensity;
		const previousBackgroundRotation = scene.backgroundRotation.clone();

		let pmrem = null;
		let spareTarget = null;
		let baked = false;
		const shadowStates = [];

		const capture = ( source, position, index ) => {

			pmrem = generator.fromScene( source, 0, near, far, { size: resolution, position, renderTarget: pmrem } );

			for ( let lod = 0; lod <= maxLod; lod ++ ) {

				const size = resolution >> lod;

				_region.min.set( 0, 0, 0 );
				_region.max.set( size, size, 6 );
				_destination.set( 0, 0, index * 6 );

				renderer.copyTextureToTexture( pmrem.texture, renderTarget.texture, _region, _destination, lod, lod );

			}

		};

		this.capturing = true;

		try {

			// Reuse camera-independent shadow maps throughout the capture.

			scene.traverse( ( object ) => {

				if ( object.castShadow === true && ( object.isPointLight === true || object.isSpotLight === true || object.isDirectionalLight === true ) ) {

					const shadow = object.shadow;

					if ( shadow.autoUpdate === true ) {

						shadowStates.push( shadow );
						shadow.autoUpdate = false;
						shadow.needsUpdate = true;

					}

				}

			} );

			this.texture = hasPrevious ? previousTexture : renderTarget.texture;
			this.bakedProbes = hasPrevious ? previousProbes : probes;
			// Distant light enters through background miss rays, not unoccluded
			// environment lighting on surfaces inside closed rooms.
			scene.background = scene.environment;
			scene.backgroundNode = null;
			scene.backgroundIntensity = scene.environmentIntensity;
			scene.backgroundRotation.copy( scene.environmentRotation );
			// Local providers report no coverage until a complete previous pass exists.
			scene.environmentNode = null;
			renderer.contextNode = context( {
				...previousContext.value,
				// Camera-space occlusion must not be sampled by a cube capture.
				getAO: null,
				getSpecularOcclusion: null,
				indirectDiffuseNode: previousContext.value.indirectDiffuseNode || indirectDiffuse( [ reflectionProbesDiffuse( this ) ] ),
				indirectSpecularNode: previousContext.value.indirectSpecularNode || indirectSpecular( [ reflectionProbesSpecular( this ) ] )
			} );

			for ( let bounce = 0; bounce <= bounces; bounce ++ ) {

				this._captureBounce = bounce + ( hasPrevious ? 1 : 0 );

				if ( bounce > 0 ) {

					// Read a complete pass while writing the next into a separate texture.
					this.texture = renderTarget.texture;
					this.bakedProbes = probes;
					if ( spareTarget === null ) {

						spareTarget = renderTarget.clone();
						renderer.initRenderTarget( spareTarget );

					}

					[ renderTarget, spareTarget ] = [ spareTarget, renderTarget ];

				}

				// Unfilled cubes stay black, so an empty environment needs no capture.

				if ( scene.environment !== null ) capture( environment, _position.set( 0, 0, 0 ), 0 );

				for ( let i = 0; i < probes.length; i ++ ) {

					capture( scene, probes[ i ].getWorldPosition( _position ), i + 1 );

				}

			}

			this.texture = renderTarget.texture;
			baked = true;

		} finally {

			for ( const shadow of shadowStates ) shadow.autoUpdate = true;

			this.capturing = false;
			this._captureBounce = 0;
			scene.environmentNode = previousEnvironmentNode;
			scene.background = previousBackground;
			scene.backgroundNode = previousBackgroundNode;
			scene.backgroundIntensity = previousBackgroundIntensity;
			scene.backgroundRotation.copy( previousBackgroundRotation );
			renderer.contextNode = previousContext;

			if ( pmrem !== null ) pmrem.dispose();
			generator.dispose();
			if ( spareTarget !== null ) spareTarget.dispose();

			if ( baked === false ) {

				renderTarget.dispose();
				this.texture = previousTexture;
				this.bakedProbes = previousProbes;

			}

		}

		if ( this._renderTarget !== null ) this._renderTarget.dispose();

		this._renderTarget = renderTarget;
		this.bakedProbes = probes;

	}

	/**
	 * Frees the GPU-related resources allocated by this instance.
	 */
	dispose() {

		if ( this._renderTarget !== null ) this._renderTarget.dispose();

		this._renderTarget = null;
		this.texture = null;

	}

}

function volume( probe ) {

	probe.getWorldScale( _size ).multiply( probe.influenceSize );

	return _size.x * _size.y * _size.z;

}

export { ReflectionProbe, ReflectionProbes };

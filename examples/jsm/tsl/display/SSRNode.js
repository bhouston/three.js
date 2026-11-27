import { Break, Continue, Fn, If, abs, bool, cos, cross, distance, dot, float, getScreenPosition, getViewPosition, int, ivec2, luminance, max, min, mix, mul, nodeObject, normalize, outputStruct, passTexture, property, reference, reflect, roughness as surfaceRoughness, screenUV, sin, sqrt, sub, texture, textureLoad, textureSize, trunc, uniform, uv, vec2, vec3, vec4, context, isBackgroundDepth, log2, projectionDepth, depthToViewZ, viewZToPerspectiveDepth, viewZToReversedPerspectiveDepth, viewZToOrthographicDepth, viewZToReversedOrthographicDepth } from 'three/tsl';
import { FloatType, HalfFloatType, LinearFilter, LinearMipmapLinearFilter, Matrix4, NodeMaterial, NearestFilter, NodeUpdateType, QuadMesh, RenderTarget, RendererUtils, Node, Vector2, Vector3 } from 'three/webgpu';
import { bindAnalyticNoise } from '../utils/RNoise.js';
import { D_GTR, ENV_RAY_LENGTH, F_Schlick, GGXVNDFPdf, GeometryTerm, getSpecularDominantFactor, ggxReflectionSample } from '../utils/SpecularHelpers.js';
import { boxBlur } from './boxBlur.js';
import { hiZ } from './HiZNode.js';
import { traceHiZ } from '../utils/SSRAdaptiveTrace.js';
import { getSSRPixelWidth, validateSSRRayDistance } from '../utils/SSRValidation.js';
import { traceLinearSSR } from '../utils/SSRLinearTrace.js';

const _quadMesh = /*@__PURE__*/ new QuadMesh();

// Texel offsets of the neighbors whose rays a pixel reuses, rotated per pixel and frame.
const REUSE_OFFSETS = [[ 1.5, 0.5 ], [ - 0.5, 1.5 ], [ - 1.5, - 0.5 ], [ 0.5, - 1.5 ]];
let _rendererState;

// Maximum ray-march step count; `quality` (0..1) scales it to a fixed per-ray count.
const MAX_STEPS = 64;

// Coarsest Hi-Z level for adaptive tracing (64-pixel cells, as in FidelityFX SSSR).
// Measured at 720p: levels above this cost iterations without skipping more.
const HIZ_MAX_LEVEL = 6;

/**
 * @typedef {Object} SSRNodeOptions
 * @property {boolean} [stochastic=false] - When `false`, traces a single mirror reflection and softens roughness with a blur pass (first-generation SSR). When `true`, varies the reflection direction per pixel with stochastic GGX rays (second-generation SSR); higher quality on rough/glossy surfaces but noisier, so it expects a temporal/spatial denoiser downstream.
 * @property {Node<float>} [metalnessNode=null] - Per-pixel metalness. With `reflectNonMetals=false`, drives the non-metal early-out.
 * @property {Node<float>} [roughnessNode=null] - Per-pixel roughness. Drives GGX sampling and the blur mip selection.
 * @property {boolean} [reflectNonMetals=false] - Only used when `stochastic=false`. When `false`, non-metallic surfaces are discarded for a noticeable performance gain and have no coverage; set `true` to also reflect dielectrics (e.g. marble, polished wood, plastic).
 * @property {boolean} [binaryRefine=false] - Sub-step binary-search refinement of detected hits. Compile-time constant (baked into the shader at construction).
 * @property {'linear'|'adaptive'} [tracing='linear'] - Adaptive mode traces a nearest-depth hierarchy and stops at the first surface a ray reaches. Requires WebGPU.
 * @property {number} [roughMinLevel=0] - Experimental, adaptive stochastic mode only: rays from surfaces rougher than 0.1 stop descending at this Hi-Z level (FidelityFX's most_detailed_mip). `0` disables it.
 * @property {boolean} [occlusionSkipping=true] - Adaptive mode only: lets rays pass behind occluders by skipping Hi-Z cells they are entirely hidden behind, as linear tracing does sample by sample. Recovers reflections of surfaces seen past an occluder at the cost of longer rays.
 * @property {?HiZNode} [hiZ=null] - Optional shared hierarchy built from the same depth texture.
 * @property {boolean} [rayReuse=false] - Stochastic mode only: resolves each pixel from its own ray and those of four neighbors.
 * @property {Camera} [camera=null] - Camera the scene is rendered with. Inferred from the color pass when omitted.
 */

/**
 * Post processing node for computing screen space reflections (SSR).
 *
 * The output is the reflected radiance premultiplied by its coverage, with the coverage in alpha.
 * It holds no BRDF and no environment: materials shade it and fill the remaining coverage, see
 * {@link SSRNode#getSpecularSource}.
 *
 * Reference: {@link https://lettier.github.io/3d-game-shaders-for-beginners/screen-space-reflection.html}
 *
 * @augments Node
 * @three_import import { ssr } from 'three/addons/tsl/display/SSRNode.js';
 */
class SSRNode extends Node {

	static get type() {

		return 'SSRNode';

	}

	/**
	 * Constructs a new SSR node.
	 *
	 * @param {Node<vec4>} colorNode - The radiance the reflections read: the beauty pass, or the previous frame's reprojected by `reprojectedBeauty()`.
	 * @param {Node<float>} depthNode - A node that represents the beauty pass's depth.
	 * @param {Node<vec3>} normalNode - A node that represents the beauty pass's unit view-space normals.
	 * @param {SSRNodeOptions} [options] - Optional inputs for material and environment data.
	 */
	constructor( colorNode, depthNode, normalNode, options = {} ) {

		super( 'vec4' );

		const {
			stochastic = false,
			metalnessNode = null,
			roughnessNode = null,
			reflectNonMetals = false,
			binaryRefine = false
		} = options;

		let camera = options.camera ?? null;

		/**
		 * When `true`, the reflection direction is varied per pixel with stochastic GGX rays
		 * (second-generation SSR). When `false`, a single mirror reflection is traced and
		 * roughness is softened with a blur pass (first-generation SSR).
		 *
		 * @type {boolean}
		 */
		this.stochastic = stochastic;

		/**
		 * The radiance the reflections read, sampled at each hit. Its alpha is the validity of the sample.
		 *
		 * @type {TextureNode}
		 */
		this.colorNode = colorNode;

		/**
		 * A node that represents the beauty pass's depth.
		 *
		 * @type {TextureNode}
		 */
		this.depthNode = depthNode;
		this._tracing = options.tracing ?? 'linear';
		if ( ! [ 'linear', 'adaptive' ].includes( this._tracing ) ) throw new Error( 'SSRNode: Invalid tracing mode.' );
		this._roughMinLevel = options.roughMinLevel ?? 0;
		this._coneTracing = options.coneTracing ?? false;
		this._coneValidate = false;
		this._hitValidation = true;
		this._rayReuse = options.rayReuse ?? false;
		this._occlusionSkipping = options.occlusionSkipping ?? true;
		this._showIterations = false;
		this._showConeMip = false;
		this._coneTaps = options.coneTaps ?? 2;
		this.hiZ = options.hiZ ?? null;
		this._ownsHiZ = false;
		if ( this._tracing === 'adaptive' && this.hiZ === null ) {

			this.hiZ = hiZ( depthNode, { maxLevel: HIZ_MAX_LEVEL } );
			this._ownsHiZ = true;

		}

		/** Adaptive cell-iteration budget, independent of quality. @type {UniformNode<int>} */
		this.adaptiveMaxSteps = uniform( 32, 'int' );
		/** Maximum linear stochastic march steps, scaled by quality. @type {UniformNode<int>} */
		this.maxSteps = uniform( MAX_STEPS, 'int' );

		/**
		 * A node that represents the beauty pass's normals.
		 *
		 * @type {Node<vec3>}
		 */
		this.normalNode = normalNode;

		/**
		 * Per-pixel metalness, used to drive the GGX reflection sampling and the non-metal
		 * early-out. When `null`, the shader treats surfaces as non-metallic.
		 *
		 * @type {?Node<float>}
		 */
		this.metalnessNode = metalnessNode;

		/**
		 * Per-pixel roughness, used to drive the GGX reflection sampling and the blur mip
		 * selection. When `null`, the shader treats surfaces as fully smooth.
		 *
		 * @type {?Node<float>}
		 */
		this.roughnessNode = roughnessNode;

		/**
		 * Only used when {@link SSRNode#stochastic} is `false`. When `false`, non-metallic
		 * surfaces are discarded for a noticeable performance gain; set `true` to also
		 * reflect dielectrics. Baked into the shader as a compile-time constant; assigning a
		 * new value recompiles the SSR material.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this._reflectNonMetals = reflectNonMetals;

		/**
		 * The `updateBeforeType` is set to `NodeUpdateType.FRAME` since the node renders
		 * its effect once per frame in `updateBefore()`.
		 *
		 * @type {string}
		 * @default 'frame'
		 */
		this.updateBeforeType = NodeUpdateType.FRAME;

		/**
		 * Controls how far a fragment can reflect. Increasing this value result in more
		 * computational overhead but also increases the reflection distance.
		 *
		 * @type {UniformNode<float>}
		 */
		this.maxDistance = uniform( 1 );

		/**
		 * Controls the cutoff between what counts as a possible reflection hit and what does not.
		 *
		 * @type {UniformNode<float>}
		 */
		this.thickness = uniform( 0.1 );

		/**
		 * Screen-edge fade width, in UV units. As a screen-space hit approaches a screen
		 * border, its coverage is faded out over this distance. `0` disables it.
		 *
		 * @type {UniformNode<float>}
		 * @default 0.2
		 */
		this.screenEdgeFade = uniform( 0.2 );

		/**
		 * Absolute luminance cap. Reflections above this are scaled down (hue preserved).
		 *
		 * @type {UniformNode<float>}
		 * @default 10
		 */
		this.maxLuminance = uniform( 10 );

		/**
		 * This parameter controls how detailed the raymarching process works.
		 * The value ranges is `[0,1]` where `1` means best quality (the maximum number
		 * of raymarching iterations/samples) and `0` means no samples at all.
		 *
		 * A quality of `0.5` is usually sufficient for most use cases. Try to keep
		 * this parameter as low as possible. Larger values result in noticeable more
		 * overhead.
		 *
		 * @type {UniformNode<float>}
		 */
		this.quality = uniform( 0.5 );

		/**
		 * Mirror bias for the stochastic GGX sampling. Concentrates the reflected rays toward
		 * the lobe's narrow (near-mirror) core, trading sampling bias for less noise.
		 * `0` samples the full VNDF lobe; values toward `1` tighten the cone. Range `[0,1]`.
		 *
		 * @type {UniformNode<float>}
		 * @default 0
		 */
		this.mirrorBias = uniform( 0 );

		/**
		 * Stochastic mode only: roughness at which the rays start to turn from their GGX sample
		 * toward the mirror direction. Rough lobes lose most of their rays off screen; a mirror ray
		 * whose cone stands for the whole lobe (see {@link SSRNode#coneMaxLevel}) fills those pixels
		 * with an approximate, blurred reflection instead.
		 *
		 * @type {UniformNode<float>}
		 * @default 1
		 */
		this.mirrorRoughnessStart = uniform( 1 );

		/**
		 * Stochastic mode only: roughness from which every ray is the mirror ray. The default
		 * of `1`, together with {@link SSRNode#mirrorRoughnessStart}, disables the transition.
		 *
		 * @type {UniformNode<float>}
		 * @default 1
		 */
		this.mirrorRoughnessEnd = uniform( 1 );

		/**
		 * Stochastic mode only: fraction of the GGX spread the rays keep once they have turned
		 * to the mirror direction. A slight jitter breaks up the blocks of the coarse mips a
		 * pure mirror ray reads. Range `[0,1]`.
		 *
		 * @type {UniformNode<float>}
		 * @default 0.1
		 */
		this.mirrorJitter = uniform( 0.1 );

		/**
		 * Stochastic mode only: the largest cosine between a ray and the direction to the camera
		 * that is still traced. Rays turned further toward the camera are skipped, and hits fade
		 * to the environment as a ray nears the threshold. `0` skips every ray heading back toward
		 * the camera, as in GPU Pro 5; `1` traces them all. Range `[0,1]`.
		 *
		 * @type {UniformNode<float>}
		 * @default 0
		 */
		this.cameraRayThreshold = uniform( 0 );

		/**
		 * Stochastic mode only: coarsest mip of the beauty pass a ray may read at its hit. Each ray
		 * reads the mip matching the footprint of the cone it stands for, which grows with roughness
		 * and hit distance, trading noise for some blur. `0` disables it. The texture behind
		 * `colorNode` must have mipmaps (`generateMipmaps` and a trilinear `minFilter`).
		 *
		 * @type {UniformNode<float>}
		 * @default 0
		 */
		this.coneMaxLevel = uniform( 0 );

		/**
		 * Number of rays the GGX lobe is assumed to be shared between (spatial and temporal reuse
		 * downstream). Each ray's cone covers `1 / ( coneSamples * pdf )` steradians, so lower
		 * values read coarser mips: less noise, more blur.
		 *
		 * @type {UniformNode<float>}
		 * @default 4
		 */
		this.coneSamples = uniform( 4 );

		/**
		 * The quality of the blur. Must be an integer in the range `[1,3]`.
		 *
		 * Baked into the blur shader as a compile-time constant so the `(size*2+1)²`
		 * sample loop unrolls; assigning a new value recompiles the blur material.
		 *
		 * @type {number}
		 * @default 2
		 */
		this._blurQuality = 2;

		/**
		 * Enables sub-step binary-search refinement of a detected hit. When on, a coarse
		 * crossing is bisected toward the exact intersection (sharper hits, less step
		 * aliasing) at the cost of extra depth samples. Baked into the shader as a
		 * compile-time constant; assigning a new value rebuilds the SSR material.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this._binaryRefine = binaryRefine;

		/**
		 * Non-linear step distribution exponent. `1` = uniform steps; `> 1` concentrates
		 * samples near the ray origin — where most short-range reflections are missed — and
		 * spaces them out toward maxDistance, as `s = (i / steps) ^ stepExponent`.
		 *
		 * Baked into the shader as a compile-time constant so `pow()` folds to a few
		 * multiplies; assigning a new value recompiles the SSR material. Only used by the
		 * stochastic reflection path.
		 *
		 * @type {number}
		 * @default 2
		 */
		this._stepExponent = 2;

		//

		if ( camera === null ) {

			if ( this.colorNode.passNode && this.colorNode.passNode.isPassNode === true ) {

				camera = this.colorNode.passNode.camera;

			} else {

				throw new Error( 'THREE.SSRNode: No camera found. ssr() requires a camera.' );

			}

		}

		/**
		 * The camera the scene is rendered with.
		 *
		 * @type {Camera}
		 */
		this.camera = camera;

		/**
		 * The spread of the blur. Automatically set when generating mips.
		 *
		 * @private
		 * @type {UniformNode<int>}
		 */
		this._blurSpread = uniform( 1 );

		/**
		 * Represents the projection matrix of the scene's camera.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._cameraProjectionMatrix = uniform( camera.projectionMatrix );

		/**
		 * Represents the inverse projection matrix of the scene's camera.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._cameraProjectionMatrixInverse = uniform( camera.projectionMatrixInverse );

		/**
		 * Represents the near value of the scene's camera.
		 *
		 * @private
		 * @type {ReferenceNode<float>}
		 */
		this._cameraNear = reference( 'near', 'float', camera );

		/**
		 * Represents the far value of the scene's camera.
		 *
		 * @private
		 * @type {ReferenceNode<float>}
		 */
		this._cameraFar = reference( 'far', 'float', camera );

		this._cameraWorldMatrix = uniform( new Matrix4().copy( camera.matrixWorld ) );
		this._cameraWorldPosition = uniform( new Vector3().copy( camera.position ) );

		this._cameraViewMatrix = uniform( new Matrix4().copy( camera.matrixWorld ) );
		this._cameraViewMatrixInverse = uniform( new Matrix4().copy( camera.matrixWorldInverse ) );

		/**
		 * The resolution of the pass.
		 *
		 * @private
		 * @type {UniformNode<vec2>}
		 */
		this._resolution = uniform( new Vector2() );

		this._noiseIndex = uniform( 0 );

		/**
		 * The render target the SSR is rendered into.
		 *
		 * @private
		 * @type {RenderTarget}
		 */
		this._ssrRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, count: 2 } );
		this._ssrRenderTarget.textures[ 0 ].name = 'SSR / Reflections';
		this._ssrRenderTarget.textures[ 1 ].name = 'SSR / Hit Distance';

		/**
		 * The render target the rays are traced into when they are reused: the radiance and ray
		 * length of each ray, and its hit position in view space with its pdf.
		 *
		 * @private
		 * @type {RenderTarget}
		 */
		this._traceRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, count: 3, minFilter: NearestFilter, magFilter: NearestFilter } );
		this._traceRenderTarget.textures[ 0 ].name = 'SSR / Trace';
		this._traceRenderTarget.textures[ 1 ].name = 'SSR / Trace Hit Distance';
		this._traceRenderTarget.textures[ 2 ].name = 'SSR / Hit';
		this._traceRenderTarget.textures[ 2 ].type = FloatType; // half floats cannot hold the pdf of a smooth surface

		/**
		 * The material that resolves the reused rays.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._resolveMaterial = new NodeMaterial();
		this._resolveMaterial.name = 'SSRNode.Resolve';

		this._resolveFn = null;
		this._hitData = null;
		this._reusesRays = false;

		/**
		 * The render target for the blurred SSR reflections.
		 *
		 * @private
		 * @type {RenderTarget}
		 */
		this._blurRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, minFilter: LinearMipmapLinearFilter, magFilter: LinearFilter } );
		this._blurRenderTarget.texture.name = 'SSR / Blur';
		this._blurRenderTarget.texture.mipmaps.push( {}, {}, {}, {}, {} );

		/**
		 * The material that is used to render the effect.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._ssrMaterial = new NodeMaterial();
		this._ssrMaterial.name = 'SSRNode.SSR';

		/**
		 * The SSR fragment `Fn` and its shared context, captured in {@link SSRNode#setup}.
		 * Re-invoking the `Fn` produces a fresh node graph, which is how the baked
		 * compile-time constants are re-applied when they change (see {@link SSRNode#_buildSSRMaterial}).
		 *
		 * @private
		 */
		this._ssrFn = null;
		this._sharedContext = null;

		/**
		 * The blur material.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._blurMaterial = new NodeMaterial();
		this._blurMaterial.name = 'SSRNode.Blur';

		/**
		 * The copy material.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._copyMaterial = new NodeMaterial();
		this._copyMaterial.name = 'SSRNode.Copy';

		/**
		 * The result of the effect is represented as a separate texture node.
		 *
		 * @private
		 * @type {PassTextureNode}
		 */
		this._textureNode = passTexture( this, this._ssrRenderTarget.textures[ 0 ] );

		/**
		 * The distance to the hit along the ray, `ENV_RAY_LENGTH` on a miss.
		 *
		 * @private
		 * @type {PassTextureNode}
		 */
		this._hitDistanceNode = passTexture( this, this._ssrRenderTarget.textures[ 1 ] );

		let blurredTextureNode = null;

		if ( this.stochastic === false && this.roughnessNode !== null ) {

			const mips = this._blurRenderTarget.texture.mipmaps.length - 1;
			const r = this.roughnessNode;
			const lod = r.mul( r ).mul( mips ).clamp( 0, mips );

			blurredTextureNode = passTexture( this, this._blurRenderTarget.texture ).level( lod );

		}

		/**
		 * Holds the blurred SSR reflections.
		 *
		 * @private
		 * @type {?PassTextureNode}
		 */
		this._blurredTextureNode = blurredTextureNode;

	}

	/**
	 * Traversal strategy. Switching modes recompiles only the SSR material.
	 * The default linear tracer and downstream output contracts are preserved.
	 * @type {'linear'|'adaptive'}
	 */
	get tracing() {

		return this._tracing;

	}

	set tracing( value ) {

		if ( ! [ 'linear', 'adaptive' ].includes( value ) ) throw new Error( 'SSRNode: Invalid tracing mode.' );
		if ( value === this._tracing ) return;
		if ( value === 'adaptive' && this.hiZ === null ) {

			this.hiZ = hiZ( this.depthNode, { maxLevel: HIZ_MAX_LEVEL } );
			this._ownsHiZ = true;

		}

		this._tracing = value;
		this._buildSSRMaterial();

	}

	/** Whether adaptive rays pass behind occluders. Changing it recompiles adaptive tracing. @type {boolean} */
	get occlusionSkipping() {

		return this._occlusionSkipping;

	}

	set occlusionSkipping( value ) {

		if ( value === this._occlusionSkipping ) return;
		this._occlusionSkipping = value;
		this._buildSSRMaterial();

	}

	/**
	 * Debug view: replaces the reflection with a heat map of trace iterations as a
	 * fraction of the budget (blue, green, red; white where the budget ran out).
	 * Changing it recompiles the SSR material.
	 * @type {boolean}
	 */
	get showIterations() {

		return this._showIterations;

	}

	set showIterations( value ) {

		if ( value === this._showIterations ) return;
		this._showIterations = value;
		this._buildSSRMaterial();

	}

	/**
	 * With `coneTracing`: `false` takes a footprint-sized Hi-Z cell as the hit without the
	 * normal and thickness tests, as the GPU Pro 5 tracer does. Fastest, but rays passing
	 * behind or beside geometry hit it. Changing it recompiles adaptive tracing.
	 *
	 * @type {boolean}
	 * @default false
	 */
	get coneValidate() {

		return this._coneValidate;

	}

	set coneValidate( value ) {

		if ( value === this._coneValidate ) return;
		this._coneValidate = value;
		this._buildSSRMaterial();

	}

	/**
	 * Stochastic mode only: resolves each pixel from its own ray and those of four neighbors,
	 * each weighted by the pixel's BRDF over the pdf the ray was sampled with, as in Stachowiak,
	 * "Stochastic Screen-Space Reflections" (Frostbite, SIGGRAPH 2015), except that a neighbor's
	 * ray direction is reused instead of its hit point. Lowers the noise of the
	 * traced result at the cost of a resolve pass. Changing it recompiles the SSR material.
	 *
	 * @type {boolean}
	 * @default false
	 */
	get rayReuse() {

		return this._rayReuse;

	}

	set rayReuse( value ) {

		if ( value === this._rayReuse ) return;
		this._rayReuse = value;
		this._buildSSRMaterial();

	}

	/**
	 * Adaptive tracing only: `false` takes the first depth crossing as the hit without the
	 * normal, thickness and distance tests, so rays passing behind geometry hit it.
	 * Changing it recompiles adaptive tracing.
	 *
	 * @type {boolean}
	 * @default true
	 */
	get hitValidation() {

		return this._hitValidation;

	}

	set hitValidation( value ) {

		if ( value === this._hitValidation ) return;
		this._hitValidation = value;
		this._buildSSRMaterial();

	}

	/**
	 * Stochastic mode only: outputs the beauty mip each hit read as a heat map (blue = 0,
	 * red = `coneMaxLevel`, black = miss). Changing it recompiles the SSR material.
	 *
	 * @type {boolean}
	 * @default false
	 */
	get showConeMip() {

		return this._showConeMip;

	}

	set showConeMip( value ) {

		if ( value === this._showConeMip ) return;
		this._showConeMip = value;
		this._buildSSRMaterial();

	}

	/**
	 * Stochastic mode only: number of beauty taps a hit reads, spread over the disc of its cone
	 * with Gaussian weights and rotated per pixel. Each tap reads the mip of its share of the
	 * disc, so more taps read finer mips. `0` reads the disc with a single tap of a coarse mip.
	 * Changing it recompiles the SSR material.
	 *
	 * @type {number}
	 * @default 2
	 */
	get coneTaps() {

		return this._coneTaps;

	}

	set coneTaps( value ) {

		if ( value === this._coneTaps ) return;
		this._coneTaps = value;
		this._buildSSRMaterial();

	}

	/**
	 * Adaptive stochastic mode only: rays test for a hit at the Hi-Z level whose cells fit inside
	 * their cone footprint (see {@link SSRNode#coneMaxLevel}), once per cell and with a thickness
	 * widened to the cell, as Hi-Z cone tracing does. Cheaper, but surfaces narrower than the
	 * footprint are hit only in proportion to their coverage. Takes precedence over `roughMinLevel`. Changing it recompiles adaptive tracing.
	 *
	 * @type {boolean}
	 * @default false
	 */
	get coneTracing() {

		return this._coneTracing;

	}

	set coneTracing( value ) {

		if ( value === this._coneTracing ) return;
		this._coneTracing = value;
		this._buildSSRMaterial();

	}

	/** Coarsest-leaf level for rough stochastic rays; `0` is off. Changing it recompiles adaptive tracing. @type {number} */
	get roughMinLevel() {

		return this._roughMinLevel;

	}

	set roughMinLevel( value ) {

		if ( value === this._roughMinLevel ) return;
		this._roughMinLevel = value;
		this._buildSSRMaterial();

	}

	/**
	 * Non-linear step distribution exponent (compile-time constant). See the backing
	 * field for details. Assigning a new value recompiles the SSR material.
	 *
	 * @type {number}
	 */
	get stepExponent() {

		return this._stepExponent;

	}

	set stepExponent( value ) {

		if ( value !== this._stepExponent ) {

			this._stepExponent = value;
			this._buildSSRMaterial();

		}

	}

	/**
	 * Blur kernel size (compile-time constant). Assigning a new value recompiles the
	 * blur material.
	 *
	 * @type {number}
	 */
	get blurQuality() {

		return this._blurQuality;

	}

	set blurQuality( value ) {

		if ( value !== this._blurQuality ) {

			this._blurQuality = value;

			// The size is baked into the boxBlur node tree, so rebuild it (recompiles the material).
			if ( this.stochastic === false ) this._buildBlurMaterial();

		}

	}

	/**
	 * Builds (or rebuilds) the blur material's node graph, baking the current
	 * {@link SSRNode#blurQuality} as the kernel size so the sample loop unrolls.
	 *
	 * @private
	 */
	_buildBlurMaterial() {

		this._blurMaterial.fragmentNode = boxBlur( texture( this._ssrRenderTarget.texture ), { size: this._blurQuality, separation: this._blurSpread } );
		this._blurMaterial.needsUpdate = true;

	}

	/**
	 * Whether sub-step binary-search hit refinement is enabled (compile-time constant).
	 * Assigning a new value rebuilds the SSR material.
	 *
	 * @type {boolean}
	 */
	get binaryRefine() {

		return this._binaryRefine;

	}

	set binaryRefine( value ) {

		if ( value !== this._binaryRefine ) {

			this._binaryRefine = value;
			this._buildSSRMaterial();

		}

	}

	/**
	 * Whether dielectrics are reflected in the non-stochastic path (compile-time constant).
	 * Assigning a new value rebuilds the SSR material.
	 *
	 * @type {boolean}
	 */
	get reflectNonMetals() {

		return this._reflectNonMetals;

	}

	set reflectNonMetals( value ) {

		if ( value !== this._reflectNonMetals ) {

			this._reflectNonMetals = value;
			this._buildSSRMaterial();

		}

	}

	/**
	 * Rebuilds the SSR material's node graph by re-invoking the fragment `Fn`, which
	 * re-bakes the compile-time constants ({@link SSRNode#binaryRefine},
	 * {@link SSRNode#stepExponent}) at their current
	 * values. A no-op until {@link SSRNode#setup} has captured the `Fn`.
	 *
	 * @private
	 */
	_buildSSRMaterial() {

		if ( this._ssrFn === null ) return;

		// the debug views return a single value, so they are traced without reuse

		this._reusesRays = this.stochastic === true && this._rayReuse === true &&
			this.showIterations === false && this.showConeMip === false && this._traceDebug !== true;

		this._hitData = this._reusesRays ? property( 'vec4' ) : null;

		const traced = this._ssrFn();

		this._ssrMaterial.contextNode = context( this._sharedContext );
		this._ssrMaterial.fragmentNode = this._reusesRays
			? outputStruct( traced, vec4( this._hitDistance ), this._hitData )
			: outputStruct( traced, vec4( this._hitDistance ) );
		this._ssrMaterial.needsUpdate = true;

		if ( this._reusesRays ) {

			this._resolveMaterial.contextNode = context( this._sharedContext );
			this._resolveMaterial.fragmentNode = outputStruct( this._resolveFn(), vec4( this._resolvedHitDistance ) );
			this._resolveMaterial.needsUpdate = true;

		}

	}

	/**
	 * Returns the result of the effect as a texture node.
	 *
	 * @return {PassTextureNode} A texture node that represents the result of the effect.
	 */
	getTextureNode() {

		return ( this.stochastic === false && this.roughnessNode !== null ) ? this._blurredTextureNode : this._textureNode;

	}

	/**
	 * Returns the reflected radiance in RGB and its coverage in alpha at the surface being shaded,
	 * e.g. as a specular source of `coverageCascade()`.
	 *
	 * @param {?Node<vec4>} [reflectionNode=null] - A filtered version of the output at the surface to use in its place, e.g. from a denoiser.
	 * @return {Node<vec4>} The radiance and coverage.
	 */
	getSpecularSource( reflectionNode = null ) {

		let reflection = reflectionNode || this._textureNode.sample( screenUV );

		if ( reflectionNode === null && this.stochastic === false && this.roughnessNode !== null ) {

			const mips = this._blurRenderTarget.texture.mipmaps.length - 1;

			reflection = passTexture( this, this._blurRenderTarget.texture ).sample( screenUV );
			reflection.levelNode = surfaceRoughness.pow2().mul( mips );

		}

		return vec4( reflection );

	}

	/**
	 * Returns the distance to the hit along the ray as a texture node, e.g. for denoisers.
	 *
	 * @return {PassTextureNode} A texture node that holds the hit distance, `ENV_RAY_LENGTH` on a miss.
	 */
	getHitDistanceNode() {

		return this._hitDistanceNode;

	}

	/**
	 * Sets the size of the effect.
	 *
	 * @param {number} width - The width of the effect.
	 * @param {number} height - The height of the effect.
	 */
	setSize( width, height ) {

		this._resolution.value.set( width, height );
		this._ssrRenderTarget.setSize( width, height );
		this._blurRenderTarget.setSize( width, height );
		if ( this._reusesRays ) this._traceRenderTarget.setSize( width, height );

	}

	/**
	 * This method is used to render the effect once per frame.
	 *
	 * @param {NodeFrame} frame - The current node frame.
	 */
	updateBefore( frame ) {

		const { renderer } = frame;

		this._cameraWorldMatrix.value.copy( this.camera.matrixWorld );
		this._cameraWorldPosition.value.copy( this.camera.position );

		_rendererState = RendererUtils.resetRendererState( renderer, _rendererState );

		const ssrRenderTarget = this._ssrRenderTarget;
		const blurRenderTarget = this._blurRenderTarget;

		const { width, height } = this.depthNode.value.image;

		_quadMesh.material = this._ssrMaterial;

		this.setSize( width, height );

		// Advance the noise index once per frame (matches SSGI / Denoise).
		this._noiseIndex.value = ( this._noiseIndex.value + 1 ) % 0x7fffffff;

		// clear

		renderer.setMRT( null );
		renderer.setClearColor( 0x000000, 0 );

		// ssr

		renderer.setRenderTarget( this._reusesRays ? this._traceRenderTarget : ssrRenderTarget );
		_quadMesh.name = 'SSR / Trace';
		_quadMesh.render( renderer );

		// resolve the reused rays (optional)

		if ( this._reusesRays ) {

			_quadMesh.material = this._resolveMaterial;

			renderer.setRenderTarget( ssrRenderTarget );
			_quadMesh.name = 'SSR / Resolve';
			_quadMesh.render( renderer );

		}

		// blur (optional)

		if ( this.stochastic === false && this.roughnessNode !== null ) {

			// blur mips but leave the base mip unblurred

			for ( let i = 0; i < blurRenderTarget.texture.mipmaps.length; i ++ ) {

				_quadMesh.material = ( i === 0 ) ? this._copyMaterial : this._blurMaterial;

				this._blurSpread.value = i;
				renderer.setRenderTarget( blurRenderTarget, 0, i );
				_quadMesh.name = `SSR / Blur / Mip ${ i }`;
				_quadMesh.render( renderer );

			}

		}

		// restore

		RendererUtils.restoreRendererState( renderer, _rendererState );

	}

	/**
	 * This method is used to setup the effect's TSL code.
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {PassTextureNode}
	 */
	setup( builder ) {

		// the depth node defines the resolution of the effect and must be updated first

		builder.getNodeProperties( this ).depthNode = this.depthNode;

		// the owned hierarchy follows the depth node, which can be swapped for a downsampled one

		if ( this._ownsHiZ && this.hiZ.depthNode !== this.depthNode ) {

			this.hiZ.dispose();
			this.hiZ = hiZ( this.depthNode, { maxLevel: HIZ_MAX_LEVEL } );

		}

		const uvNode = uv();

		const pointToLineDistance = Fn( ( [ point, linePointA, linePointB ] ) => {

			// https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

			return cross( point.sub( linePointA ), point.sub( linePointB ) ).length().div( linePointB.sub( linePointA ).length() );

		}, { point: 'vec3', linePointA: 'vec3', linePointB: 'vec3', return: 'float' } );

		const pointPlaneDistance = Fn( ( [ point, planePoint, planeNormal ] ) => {

			// https://mathworld.wolfram.com/Point-PlaneDistance.html
			// https://en.wikipedia.org/wiki/Plane_(geometry)
			// http://paulbourke.net/geometry/pointlineplane/

			// planeNormal is already normalized, so the denominator is 1.
			const d = mul( planeNormal.x, planePoint.x ).add( mul( planeNormal.y, planePoint.y ) ).add( mul( planeNormal.z, planePoint.z ) ).negate().toVar();
			const distance = mul( planeNormal.x, point.x ).add( mul( planeNormal.y, point.y ) ).add( mul( planeNormal.z, point.z ) ).add( d );
			return distance;

		}, { point: 'vec3', planePoint: 'vec3', planeNormal: 'vec3', return: 'float' } );

		const sampleDepth = ( uv ) => projectionDepth( this.depthNode.sample( uv ).r, this._cameraNear, this._cameraFar );

		const sampleMarchNoise = this.stochastic === true ? bindAnalyticNoise( this._resolution, 47 ) : null;

		const computeScreenBorderFactor = Fn( ( [ uvCoord, borderWidth ] ) => {

			const border = borderWidth.max( 1e-4 );

			// Distance to the nearest screen edge — uniform falloff at corners.
			const edgeDist = min(
				min( uvCoord.x, float( 1 ).sub( uvCoord.x ) ),
				min( uvCoord.y, float( 1 ).sub( uvCoord.y ) )
			);

			// Two smoothsteps for a softer ease-in-out than a single ramp.
			const t = edgeDist.smoothstep( 0, border );

			return t.smoothstep( 0, 1 ).pow( 0.125 );

		}, { uvCoord: 'vec2', borderWidth: 'float', return: 'float' } );

		const hitDistance = this._hitDistance = property( 'float' );

		const ssr = Fn( () => {

			const adaptive = this.tracing === 'adaptive';
			if ( adaptive && builder.renderer.backend.isWebGPUBackend !== true ) throw new Error( 'SSRNode: Adaptive tracing requires WebGPU.' );

			const noise = this.stochastic === true ? sampleMarchNoise( uvNode, this._noiseIndex ) : null;

			const depth = sampleDepth( uvNode ).toVar();

			// Skip background pixels (cleared far-plane depth); the target is cleared each frame.
			isBackgroundDepth( depth ).discard();

			const viewPosition = getViewPosition( uvNode, depth, this._cameraProjectionMatrixInverse ).toVar();
			const worldPosition = this._cameraWorldMatrix.mul( vec4( viewPosition, 1.0 ) ).xyz.toVar();
			const viewNormal = this.normalNode.rgb.toVar();

			const viewIncidentDir = ( ( this.camera.isPerspectiveCamera ) ? normalize( viewPosition ) : vec3( 0, 0, - 1 ) ).toVar();

			// The node system samples the metalness/roughness textures at the current uv,
			// so no explicit sample() is needed here.
			const metalness = float( this.metalnessNode );

			if ( this.stochastic === false && this._reflectNonMetals === false ) {

				metalness.lessThanEqual( 0.0 ).discard();

			}

			const roughness = float( this.roughnessNode );
			const glossiness = min( roughness.div( 0.25 ), 1 ).oneMinus();
			const hitBorderWidth = this.screenEdgeFade.mul( glossiness );

			const V = viewIncidentDir.negate().normalize().toVar();

			let viewReflectDir, specDominantFactor;
			let coneTangent = null, traceTangent = null;

			if ( this.stochastic === false ) {

				viewReflectDir = reflect( viewIncidentDir, viewNormal ).normalize().toVar();
				specDominantFactor = float( 1 );

			} else {

				const Xi = noise.toVar();
				// mirror bias trades the unbiased GGX lobe for lower variance
				Xi.y.assign( mix( Xi.y, 0.0, this.mirrorBias.mul( Xi.w.sqrt() ) ) );

				const ggxSample = ggxReflectionSample( viewNormal, V, roughness, metalness, vec3( 1 ), Xi ).toVar();

				// Sometimes the GGX sample is facing away from the surface, so we need to re-sample.
				If( ggxSample.get( 'reflectDir' ).dot( viewNormal ).lessThan( 0 ), () => {

					ggxSample.assign( ggxReflectionSample( viewNormal, V, roughness, metalness, vec3( 1 ), Xi.add( Xi.mul( 7 ) ).fract() ) );

				} );

				// Rough lobes turn into the mirror ray, which fills more pixels than rays lost off screen.
				// ponytail: the mirror direction, not the lobe's dominant direction (which leans to the normal when rough).
				const mirror = roughness.sub( this.mirrorRoughnessStart ).div( this.mirrorRoughnessEnd.sub( this.mirrorRoughnessStart ).max( 1e-4 ) ).clamp().mul( this.mirrorJitter.oneMinus() ).toVar();

				viewReflectDir = mix( ggxSample.get( 'reflectDir' ), reflect( viewIncidentDir, viewNormal ), mirror ).normalize().toVar();

				// the pdf of the direction that is traced, which ray reuse weights it by
				const pdf = GGXVNDFPdf( dot( viewNormal, normalize( V.add( viewReflectDir ) ) ).max( 0 ), ggxSample.get( 'NdotV' ), ggxSample.get( 'alpha' ) ).toVar();

				// Cone footprint (filtered importance sampling): the ray stands for
				// 1 / ( coneSamples * pdf ) steradians of the lobe, and a mirror ray for all of it.
				coneTangent = pdf.mul( mix( this.coneSamples, 1, mirror ) ).mul( Math.PI ).max( 1e-6 ).inverseSqrt().toVar();

				// A mirror ray is traced as a thin ray: hits quantized to Hi-Z cells of its wide cone show as blocks.
				traceTangent = coneTangent.mul( mirror.oneMinus() ).toVar();
				specDominantFactor = getSpecularDominantFactor( ggxSample.get( 'NdotV' ), roughness ).toVar();

				// a miss is stored as a hit far along the ray, so a neighbor sees it in the same direction

				if ( this._reusesRays ) this._hitData.assign( vec4( viewPosition.add( viewReflectDir.mul( ENV_RAY_LENGTH ) ), pdf ) );

			}

			// The margin of the ray below the camera ray threshold: negative when it turns too far toward the camera.
			const cameraFacing = this.stochastic ? this.cameraRayThreshold.sub( dot( viewReflectDir, V ) ).toVar() : null;

			const maxReflectRayLen = this.maxDistance.div( dot( viewIncidentDir.negate(), viewNormal ).max( 1e-4 ) ).toVar();

			const d1viewPosition = viewPosition.add( viewReflectDir.mul( maxReflectRayLen ) ).toVar();

			// Camera type is fixed at build time, so guard the near-plane clamp with a JS branch
			// rather than a runtime uniform (the orthographic case compiles it out entirely).
			if ( this.camera.isPerspectiveCamera ) {

				If( d1viewPosition.z.greaterThan( this._cameraNear.negate() ), () => {

					const t = sub( this._cameraNear.negate(), viewPosition.z ).div( viewReflectDir.z );
					d1viewPosition.assign( viewPosition.add( viewReflectDir.mul( t ) ) );

				} );

			}

			const d0 = uvNode.mul( this._resolution ).xy.toVar();
			const d1 = getScreenPosition( d1viewPosition, this._cameraProjectionMatrix ).mul( this._resolution ).toVar();

			// clip the ray to the screen so all steps land on visible texels

			const extent = d1.sub( d0 ).toVar();
			const exitT = vec2(
				extent.x.greaterThanEqual( 0 ).select( this._resolution.x.sub( d0.x ), d0.x ),
				extent.y.greaterThanEqual( 0 ).select( this._resolution.y.sub( d0.y ), d0.y )
			).div( extent.abs().max( 1e-6 ) );
			const clipT = min( exitT.x, exitT.y ).min( 1 ).toConst();

			If( clipT.lessThan( 1 ), () => {

				d1.assign( d0.add( extent.mul( clipT ) ) );
				d1viewPosition.assign( this.camera.isPerspectiveCamera
					? mix( viewPosition.div( viewPosition.z ), d1viewPosition.div( d1viewPosition.z ), clipT ).div( mix( viewPosition.z.reciprocal(), d1viewPosition.z.reciprocal(), clipT ) )
					: mix( viewPosition, d1viewPosition, clipT ) );

			} );

			const xLen = d1.x.sub( d0.x ).toVar();
			const yLen = d1.y.sub( d0.y ).toVar();

			// dominant-axis ray length in texels (used for the per-step floor below)
			const rayLen = max( xLen.abs(), yLen.abs() ).max( 1 ).toVar();

			// Blur traces a single mirror ray, so spend steps in proportion to the ray's screen-space
			// length (cheap for the short rays that dominate). Scatter needs a fixed, bounded count for
			// coherent stochastic sampling; each step then spans the whole ray as rayVec / totalStep.
			const totalStep = int( this.stochastic === false
				? trunc( max( abs( xLen ), abs( yLen ) ).mul( this.quality.clamp() ) ).max( int( 1 ) ).toConst()
				: this.quality.clamp().mul( this.maxSteps.clamp( 1, 4096 ) ).max( float( 1 ) ) ).toConst();

			const xSpan = xLen.div( totalStep ).toVar();
			const ySpan = yLen.div( totalStep ).toVar();

			const stepVec = vec2( xSpan, ySpan ).toVar();
			const invResolution = vec2( float( 1 ), float( 1 ) ).div( this._resolution ).toVar();
			const uvPixelStepX = vec2( invResolution.x, float( 0 ) ).toVar();

			const output = vec4( 0 ).toVar();
			const hit = float( 0 ).toVar();
			hitDistance.assign( ENV_RAY_LENGTH );

			// Reflected-ray view-space Z at ray parameter s ∈ [0,1] (linear in 1/z for perspective),
			// hoisted so the march and refinement evaluate it identically.
			const recipVPZ = float( 1 ).div( viewPosition.z ).toConst();
			const recipD1VPZ = float( 1 ).div( d1viewPosition.z ).toConst();

			// Camera type is known at build time, so branch at compile time rather than via a runtime select.
			const reflectRayZAt = this.camera.isPerspectiveCamera
				? ( sVal ) => float( 1 ).div( recipVPZ.add( sVal.mul( recipD1VPZ.sub( recipVPZ ) ) ) )
				: ( sVal ) => viewPosition.z.add( sVal.mul( d1viewPosition.z.sub( viewPosition.z ) ) );

			// view-space point on the reflected ray at s

			const rayPointAt = this.camera.isPerspectiveCamera
				? ( sVal ) => mix( viewPosition.div( viewPosition.z ), d1viewPosition.div( d1viewPosition.z ), sVal ).mul( reflectRayZAt( sVal ) )
				: ( sVal ) => mix( viewPosition, d1viewPosition, sVal );

			// Screen-space position along the ray for a given s ∈ [0,1].
			const screenPosAt = ( sVal ) => d0.add( stepVec.mul( sVal.mul( totalStep ) ) );

			// Ray parameter s ∈ [0,1] for step `idx`. Blur marches uniformly (matching the original loop:
			// one ~texel step per iteration). Scatter uses an exponential remap `(idx/steps)^stepExponent`
			// that concentrates samples near the origin, floored to ≥1 texel/step; `jitter` dissolves banding.
			const sampleFraction = this.stochastic === false
				? ( idx ) => idx.div( totalStep )
				: ( idx ) => max(
					idx.add( noise.z.sub( 0.5 ) ).div( totalStep ).pow( this.stepExponent ),
					idx.div( rayLen )
				);

			const foundHit = bool( false ).toVar();
			// The ray reached a surface it cannot shade, e.g. the back of one next to its origin: it is not free to see the environment.
			const blocked = bool( false ).toVar();
			const hitUvS = vec2( 0 ).toVar();
			const hitD = float( 0 ).toVar();
			const validatedPosition = vec3( 0 ).toVar();
			const adaptiveRay = d1viewPosition.sub( viewPosition ).toConst();
			const surfacePoint = ( uv, raw ) => getViewPosition( uv, projectionDepth( raw, this._cameraNear, this._cameraFar ), this._cameraProjectionMatrixInverse );
			// `cell` widens the tolerance to a coarse Hi-Z cell's size in pixels (cone tracing).
			const nearRay = ( point, cell = 0 ) => {

				const pixelWidth = getSSRPixelWidth( this._cameraProjectionMatrixInverse, point.z, invResolution.x, this.camera.isPerspectiveCamera );
				return validateSSRRayDistance( point, viewPosition, adaptiveRay, max( pixelWidth.mul( float( cell ).add( 3 ) ), this.thickness ) );

			};

			const iterationHeat = ( fraction, exhausted ) => {

				const cool = mix( vec3( 0, 0, 1 ), vec3( 0, 1, 0 ), fraction.mul( 2 ).clamp() );
				const warm = mix( cool, vec3( 1, 0, 0 ), fraction.mul( 2 ).sub( 1 ).clamp() );
				return vec4( bool( exhausted ).select( vec3( 1 ), warm ), 1 );

			};

			const validateHit = ( hitUV, raw, cell ) => {

				const valid = bool( false ).toVar();
				const allowed = dot( viewReflectDir, this.normalNode.sample( hitUV ).rgb ).lessThan( 0 );
				If( allowed, () => {

					const point = surfacePoint( hitUV, raw ).toVar();
					validatedPosition.assign( point );
					valid.assign( nearRay( point, cell ) );

					if ( this.stochastic === false ) valid.assign( valid.and( pointPlaneDistance( point, viewPosition, viewNormal ).lessThanEqual( this.maxDistance ) ) );

				} );
				return valid;

			};

			if ( adaptive ) {

				const reversed = builder.renderer.reversedDepthBuffer === true && builder.renderer.logarithmicDepthBuffer !== true;
				const toDepth = this.camera.isPerspectiveCamera
					? reversed ? viewZToReversedPerspectiveDepth : viewZToPerspectiveDepth
					: reversed ? viewZToReversedOrthographicDepth : viewZToOrthographicDepth;
				const result = traceHiZ( this.hiZ,
					vec3( uvNode, toDepth( viewPosition.z, this._cameraNear, this._cameraFar ) ),
					vec3( d1.mul( invResolution ), toDepth( d1viewPosition.z, this._cameraNear, this._cameraFar ) ), {
						reversedDepth: reversed,
						start: cameraFacing !== null ? cameraFacing.lessThan( 0 ).select( 1, rayLen.reciprocal() ) : rayLen.reciprocal(),
						maxSteps: this.adaptiveMaxSteps,
						// Cone tracing: cells no larger than the footprint radius, taking pixels travelled
						// on screen as the distance (exact for rays parallel to the screen).
						minLevel: this.stochastic && this.coneTracing ? travelled => int( log2( travelled.mul( traceTangent ).max( 1 ) ).min( this.coneMaxLevel ) )
							: this.stochastic && this.roughMinLevel > 0 ? roughness.greaterThan( 0.1 ).select( int( this.roughMinLevel ), int( 0 ) ) : 0,
						coarseHits: this.stochastic && this.coneTracing,
						depthTolerance: 2e-7,
						projectDepth: raw => projectionDepth( raw, this._cameraNear, this._cameraFar ),
						// The cell's farthest surface fails the thickness test where the ray
						// enters and leaves, so every nearer surface in between fails it too.
						isHidden: this.occlusionSkipping === false ? null : ( uvIn, uvOut, raw ) => nearRay( surfacePoint( uvIn, raw ).toVar() ).not().and( nearRay( surfacePoint( uvOut, raw ).toVar() ).not() ),
						acceptHit: ( uv, s, raw, cell ) => {

							if ( this.hitValidation && ( cell === undefined || this.coneValidate ) ) return validateHit( uv, raw, cell );
							// Unvalidated hit: shade a coarse one at the ray's own point.
							validatedPosition.assign( cell === undefined ? surfacePoint( uv, raw ) : rayPointAt( s ) );
							return bool( true );

						}
					} );
				if ( this._traceDebug === true ) return vec4( float( result.iterations ), 0, float( result.hit ), float( result.exhausted ) );
				if ( this.showIterations ) return iterationHeat( float( result.iterations ).div( float( this.adaptiveMaxSteps ) ), result.exhausted );
				foundHit.assign( result.hit );
				blocked.assign( result.rejected );
				hitUvS.assign( result.uv );
				hitD.assign( projectionDepth( result.depth, this._cameraNear, this._cameraFar ) );

			} else {

				const result = traceLinearSSR( {
					totalStep,
					sampleFraction,
					uvAt: s => screenPosAt( s ).mul( invResolution ),
					sampleDepth: uv => this.depthNode.sample( uv ).r,
					projectDepth: raw => projectionDepth( raw, this._cameraNear, this._cameraFar ),
					sceneZ: raw => depthToViewZ( raw, this._cameraNear, this._cameraFar, this.camera ),
					rayZAt: reflectRayZAt,
					isBackground: isBackgroundDepth,
					binaryRefine: this.binaryRefine,
					frontFacing: uv => dot( viewReflectDir, this.normalNode.sample( uv ).rgb ).lessThan( 0 ),
					acceptHit: ( uvS, s, lastS, d ) => {

						const valid = bool( false ).toVar();
						const vP = getViewPosition( uvS, d, this._cameraProjectionMatrixInverse ).toVar();
						const away = pointToLineDistance( vP, viewPosition, d1viewPosition ).toVar();
						const neighbor = getViewPosition( uvS.add( uvPixelStepX ), d, this._cameraProjectionMatrixInverse );
						const minThickness = neighbor.x.sub( vP.x ).mul( 3 );
						const stepLength = distance( rayPointAt( lastS ), rayPointAt( s ) );
						const tk = this.binaryRefine ? max( max( minThickness, this.thickness ), stepLength ) : max( minThickness, this.thickness );
						If( away.lessThanEqual( tk ), () => {

							If( dot( viewReflectDir, this.normalNode.sample( uvS ).rgb ).greaterThanEqual( 0 ), () => {

								Continue();

							} );
							if ( this.stochastic === false ) {

								If( pointPlaneDistance( vP, viewPosition, viewNormal ).greaterThan( this.maxDistance ), () => {

									Break();

								} );

							}

							valid.assign( true );

						} );
						return valid;

					}
				} );
				if ( this.showIterations ) return iterationHeat( float( result.iterations ).div( float( totalStep ) ), false );
				foundHit.assign( result.hit );
				hitUvS.assign( result.uv );
				hitD.assign( result.depth );

			}

			const coneLod = float( 0 ).toVar();

			If( foundHit, () => {

				// Shade the hit, reusing the depth fetched during the march (or refinement).
				const uvS = hitUvS;
				const vP = ( adaptive ? validatedPosition : getViewPosition( uvS, hitD, this._cameraProjectionMatrixInverse ) ).toVar();

				// In blur mode the ratio² falloff re-grows past maxDistance, so over-range hits fall back
				// to env. The scatter path bounds reach via ray length, so every hit shades.
				const distancePointPlane = this.stochastic === false ? pointPlaneDistance( vP, viewPosition, viewNormal ).toVar() : float( 0 );
				const withinRange = distancePointPlane.lessThanEqual( this.maxDistance ).toVar();

				if ( this.binaryRefine && ! adaptive ) {

					// bisection can land on a depth discontinuity, so validate the refined hit

					const neighbor = getViewPosition( uvS.add( uvPixelStepX ), hitD, this._cameraProjectionMatrixInverse );
					const tk = max( neighbor.x.sub( vP.x ).mul( 3 ), this.thickness );
					withinRange.assign( withinRange.and( isBackgroundDepth( hitD ).not() )
						.and( pointToLineDistance( vP, viewPosition, d1viewPosition ).lessThanEqual( tk ) ) );

					withinRange.assign( withinRange.and( dot( viewReflectDir, this.normalNode.sample( uvS ).rgb ).lessThan( 0 ) ) );

				}

				If( withinRange, () => {

					const hitWorldPosition = this._cameraWorldMatrix.mul( vec4( vP, 1.0 ) ).xyz.toVar();
					const worldDistance = distance( worldPosition, hitWorldPosition ).mul( specDominantFactor ).toVar();

					let reflectSample = this.colorNode.sample( uvS );

					if ( this.stochastic === true ) {

						// Read the beauty mip covering the cone's disc at the hit.
						// ponytail: isotropic, not depth-aware; use a depth-aware Gaussian pyramid if it leaks.
						const radius = distance( viewPosition, vP ).mul( coneTangent );
						const pixelSize = getViewPosition( uvS.add( uvPixelStepX ), hitD, this._cameraProjectionMatrixInverse ).x.sub( vP.x ).abs().max( 1e-6 );
						// `pixelSize` is a texel of the pass, the mips are those of the beauty pass
						const colorScale = float( textureSize( this.colorNode ).x ).div( this._resolution.x );
						const diameter = radius.mul( 2 ).div( pixelSize ).toVar(); // in pixels of the pass
						coneLod.assign( log2( diameter.mul( colorScale ) ).clamp( 0, this.coneMaxLevel ) );
						reflectSample = reflectSample.level( coneLod );

						if ( this.coneTaps > 0 ) {

							// A Gaussian over the disc, as a spiral of taps that each read the mip of their share of it.

							const taps = this.coneTaps;
							const tapLod = log2( diameter.mul( colorScale ).div( Math.sqrt( taps ) ) ).clamp( 0, this.coneMaxLevel ).toVar();
							const rotation = noise.w.mul( 2 * Math.PI ).toVar();
							const sum = vec4( 0 ).toVar();
							let total = 0;

							for ( let i = 0; i < taps; i ++ ) {

								const r = Math.sqrt( ( i + 0.5 ) / taps );
								const weight = Math.exp( - 2 * r * r );
								const angle = rotation.add( i * 2.399963 );
								const offset = vec2( cos( angle ), sin( angle ) ).mul( diameter.mul( 0.5 * r ) ).div( this._resolution );

								sum.addAssign( this.colorNode.sample( uvS.add( offset ) ).level( tapLod ).mul( weight ) );
								total += weight;

							}

							reflectSample = sum.div( total );

						}

					}

					const reflectColor = reflectSample.toVar();
					// the coverage fades out at the screen borders and, for the mirror path, over the distance

					let coverage = reflectColor.a.mul( computeScreenBorderFactor( uvS, this.stochastic === true ? hitBorderWidth : this.screenEdgeFade ) );

					if ( this.stochastic === false ) coverage = coverage.mul( distancePointPlane.div( this.maxDistance ).oneMinus().pow2() );

					// it also fades out as the ray nears the camera ray threshold; a threshold of 1 keeps every ray

					if ( cameraFacing !== null ) coverage = coverage.mul( this.cameraRayThreshold.greaterThanEqual( 1 ).select( 1, cameraFacing.mul( 4 ).clamp() ) );

					// the point on the ray at the hit distance: the validated surface point of a short ray can lie behind the pixel

					if ( this._reusesRays ) this._hitData.xyz.assign( viewPosition.add( viewReflectDir.mul( distance( viewPosition, vP ) ) ) );

					hit.assign( 1 );
					output.assign( vec4( reflectColor.rgb, 1 ).mul( coverage ) );
					hitDistance.assign( worldDistance );

				} );

			} );

			// A blocked ray is not free to see the environment: it returns the beauty at its origin with full
			// coverage, and ray reuse skips it and fills the pixel from its neighbors.

			If( blocked, () => {

				output.assign( vec4( this.colorNode.sample( uvNode ).level( 0 ).rgb, 1 ) );

				if ( this._reusesRays ) this._hitData.w.assign( 0 );

			} );

			if ( this.stochastic && this.showConeMip ) return iterationHeat( coneLod.div( this.coneMaxLevel.max( 1 ) ), false ).mul( hit );

			const lum = luminance( output.rgb ).max( 1e-4 ).toVar();
			output.rgb.mulAssign( this.maxLuminance.div( lum ).min( 1 ) );

			return output.max( 0 );

		} );

		// Ray reuse, Stachowiak 2015: each ray is weighted by the BRDF of the pixel that reuses it over the
		// pdf it was sampled with, and the sum is normalized by the weights (a ratio estimator). The result
		// stays radiance premultiplied by coverage: the material applies its BRDF.

		const resolvedHitDistance = this._resolvedHitDistance = property( 'float' );

		const resolve = Fn( () => {

			const depth = sampleDepth( uvNode ).toVar();

			isBackgroundDepth( depth ).discard();

			const traceColor = texture( this._traceRenderTarget.textures[ 0 ] );
			const traceHit = texture( this._traceRenderTarget.textures[ 2 ] );

			const viewPosition = getViewPosition( uvNode, depth, this._cameraProjectionMatrixInverse ).toVar();
			const N = this.normalNode.rgb.toVar();
			const V = ( this.camera.isPerspectiveCamera ? normalize( viewPosition ).negate() : vec3( 0, 0, 1 ) ).toVar();
			const NdotV = dot( N, V ).max( 1e-4 ).toVar();

			const roughness = float( this.roughnessNode ).toVar();
			const alpha = roughness.mul( roughness ).max( 0.001 ).toVar();

			const coord = ivec2( uvNode.mul( this._resolution ) ).toVar();
			const last = ivec2( this._resolution ).sub( 1 ).toVar();
			const own = textureLoad( traceColor, coord ).toVar();

			resolvedHitDistance.assign( textureLoad( texture( this._traceRenderTarget.textures[ 1 ] ), coord ).r );

			const angle = bindAnalyticNoise( this._resolution, 71 )( uvNode, this._noiseIndex ).x.mul( 2 * Math.PI ).toVar();
			const rotation = vec2( cos( angle ), sin( angle ) ).toVar();

			// the pdf of the bounded VNDF sampler, see `ggxReflectionSample()`

			const sinV2 = NdotV.mul( NdotV ).oneMinus().max( 0 ).toVar();
			const s2 = sqrt( sinV2 ).add( 1 ).pow2().toVar();
			const a2 = alpha.mul( alpha ).toVar();
			const k = a2.oneMinus().mul( s2 ).div( s2.add( a2.mul( NdotV ).mul( NdotV ) ) );
			const pdfScale = k.mul( NdotV ).add( sqrt( a2.mul( sinV2 ).add( NdotV.mul( NdotV ) ) ) ).mul( 2 ).toVar();

			// the weight only needs how the Fresnel term varies across the lobe, not its tint

			const f0 = vec3( mix( float( 0.04 ), float( 1 ), float( this.metalnessNode ?? 1 ) ) ).toVar();

			const pixelWidth = getSSRPixelWidth( this._cameraProjectionMatrixInverse, viewPosition.z, this._resolution.x.reciprocal(), this.camera.isPerspectiveCamera ).toVar();

			const sum = vec4( 0 ).toVar();
			const total = float( 0 ).toVar();

			const accumulate = ( texel, isOwn ) => {

				const hit = textureLoad( traceHit, texel ).toVar();

				// The direction of the ray is reused, not its hit point: aiming at a hit that is as near as
				// the neighbor itself changes the direction, and with it the weight, by a large amount.

				let origin = viewPosition, sameSurface = null;

				if ( isOwn === false ) {

					const neighborUV = vec2( texel ).add( 0.5 ).div( this._resolution ).toVar();

					origin = getViewPosition( neighborUV, sampleDepth( neighborUV ), this._cameraProjectionMatrixInverse ).toVar();

					// a neighbor on another surface sees other things

					sameSurface = dot( N, this.normalNode.sample( neighborUV ).rgb ).greaterThan( 0.9 )
						.and( dot( N, origin.sub( viewPosition ) ).abs().lessThan( pixelWidth.mul( 2 ) ) );

				}

				const L = normalize( hit.xyz.sub( origin ) ).toVar();
				const NdotL = dot( N, L ).toVar();

				// a ray is usable when it was traced and leaves the surface of this pixel

				let usable = hit.w.greaterThan( 0 ).and( NdotL.greaterThan( 0 ) );

				if ( isOwn === false ) usable = usable.and( texel.notEqual( coord ).any() ).and( sameSurface );

				If( usable, () => {

					const geometry = GeometryTerm( NdotL, NdotV, alpha ).div( NdotV.mul( 4 ) );

					// the distribution cancels against the pdf of the pixel's own ray, which keeps a mirror stable

					const H = normalize( V.add( L ) ).toVar();

					const lobe = isOwn
						? geometry.mul( pdfScale )
						: D_GTR( alpha, dot( N, H ).max( 0 ), float( 2 ) ).mul( geometry ).div( hit.w );

					// the Fresnel term varies across the lobe of a dielectric, so it is part of the weight

					const weight = F_Schlick( f0, dot( V, H ).max( 0 ), float( 1 ) ).x.mul( lobe ).toVar();

					sum.addAssign( textureLoad( traceColor, texel ).mul( weight ) );
					total.addAssign( weight );

				} );

			};

			accumulate( coord, true );

			for ( const [ x, y ] of REUSE_OFFSETS ) {

				const offset = vec2( rotation.x.mul( x ).sub( rotation.y.mul( y ) ), rotation.y.mul( x ).add( rotation.x.mul( y ) ) );

				accumulate( coord.add( ivec2( offset.round() ) ).clamp( ivec2( 0 ), last ).toVar(), false );

			}

			// without a usable ray the pixel keeps its own result, which is the beauty at its origin with full coverage for a blocked ray

			return total.greaterThan( 0 ).select( sum.div( total.max( 1e-6 ) ), own );

		} );

		this._ssrFn = ssr;
		this._resolveFn = resolve;
		this._sharedContext = builder.getSharedContext();
		this._buildSSRMaterial();

		const reflectionBuffer = texture( this._ssrRenderTarget.texture );

		if ( this.stochastic === false ) {

			this._buildBlurMaterial();

		}

		this._copyMaterial.fragmentNode = reflectionBuffer;
		this._copyMaterial.needsUpdate = true;

		//

		return this.getTextureNode();

	}

	getRenderTarget() {

		return this._ssrRenderTarget;

	}

	/**
	 * Frees internal resources. This method should be called
	 * when the effect is no longer required.
	 */
	dispose() {

		super.dispose();

		if ( this._ownsHiZ ) this.hiZ.dispose();

		this._ssrRenderTarget.dispose();
		this._traceRenderTarget.dispose();
		this._blurRenderTarget.dispose();

		this._ssrMaterial.dispose();
		this._resolveMaterial.dispose();
		this._blurMaterial.dispose();
		this._copyMaterial.dispose();

	}

}

export default SSRNode;

/**
 * TSL function for creating screen space reflections (SSR).
 *
 * @tsl
 * @function
 * @param {Node<vec4>} colorNode - The radiance the reflections read: the beauty pass, or the previous frame's reprojected by `reprojectedBeauty()`.
 * @param {Node<float>} depthNode - A node that represents the beauty pass's depth.
 * @param {Node<vec3>} normalNode - A node that represents the beauty pass's unit view-space normals.
 * @param {SSRNodeOptions} [options] - Optional inputs for material and environment data.
 * @returns {SSRNode}
 */
export const ssr = ( colorNode, depthNode, normalNode, options = {} ) => new SSRNode(
	nodeObject( colorNode ),
	nodeObject( depthNode ),
	nodeObject( normalNode ),
	options
);

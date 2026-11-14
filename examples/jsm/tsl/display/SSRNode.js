import { Break, Continue, Fn, If, abs, bool, cross, distance, div, dot, float, getScreenPosition, getViewPosition, int, luminance, max, min, mix, mul, nodeObject, normalize, passTexture, reference, reflect, sub, texture, textureSize, trunc, uniform, uv, vec2, vec3, vec4, context, isBackgroundDepth, log2, projectionDepth, depthToViewZ, viewZToPerspectiveDepth, viewZToReversedPerspectiveDepth, viewZToOrthographicDepth, viewZToReversedOrthographicDepth } from 'three/tsl';
import { HalfFloatType, LinearFilter, LinearMipmapLinearFilter, Matrix4, NodeMaterial, NodeUpdateType, QuadMesh, RenderTarget, RendererUtils, Node, Vector2, Vector3 } from 'three/webgpu';
import { bindAnalyticNoise } from '../utils/RNoise.js';
import { ENV_RAY_LENGTH, getSpecularDominantFactor, ggxReflectionSample } from '../utils/SpecularHelpers.js';
import { boxBlur } from './boxBlur.js';
import ImportanceSampledEnvironment from './ImportanceSampledEnvironment.js';
import { hiZ } from './HiZNode.js';
import { traceHiZ } from '../utils/SSRTrace.js';
import { getSSRPixelWidth, validateSSRRayDistance } from '../utils/SSRValidation.js';
import { traceLinearSSR } from '../utils/SSRLinearTrace.js';

const _quadMesh = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();
let _rendererState;

// Maximum ray-march step count; `quality` (0..1) scales it to a fixed per-ray count.
const MAX_STEPS = 64;

// Coarsest Hi-Z level for adaptive tracing (64-pixel cells, as in FidelityFX SSSR).
// Measured at 720p: levels above this cost iterations without skipping more.
const HIZ_MAX_LEVEL = 6;

/**
 * @typedef {Object} SSRNodeOptions
 * @property {boolean} [stochastic=false] - When `false`, traces a single mirror reflection and softens roughness with a blur pass (first-generation SSR). When `true`, varies the reflection direction per pixel with stochastic GGX rays (second-generation SSR); higher quality on rough/glossy surfaces but noisier, so it expects a temporal/spatial denoiser downstream.
 * @property {Node<float>} [metalnessNode=null] - Per-pixel metalness. Drives GGX reflection sampling and, with `reflectNonMetals=false`, the non-metal early-out.
 * @property {Node<vec3>} [specularNode=null] - Stochastic mode only: material normal-incidence reflectance (F0), typically an MRT `specularColorBlended` attachment (it already includes IOR, specular intensity and metalness; raw `material.specularColor` reflects too brightly). Defaults to the metalness/base-color approximation.
 * @property {Node<float>} [specularF90Node=null] - Stochastic mode only: material grazing reflectance, typically an MRT `specularF90` attachment. Defaults to one.
 * @property {Node<float>} [roughnessNode=null] - Per-pixel roughness. Drives GGX sampling and the blur mip selection.
 * @property {boolean} [reflectNonMetals=false] - Only used when `stochastic=false`. When `false`, non-metallic surfaces are discarded for a noticeable performance gain; set `true` to also reflect dielectrics (e.g. marble, polished wood, plastic).
 * @property {Texture} [environmentNode=null] - Equirectangular HDR environment map with CPU-side `image.data` (e.g. from RGBELoader). Not compatible with PMREM / `scene.environment` cubemaps.
 * @property {boolean} [envImportanceSampling=false] - When `true`, precomputes env-luminance CDF tables and uses MIS for environment misses. Build-time only.
 * @property {Node} [diffuseNode=null] - Scene diffuse / base color. Defaults to `vec3(1)` in the shader when omitted.
 * @property {boolean} [binaryRefine=false] - Sub-step binary-search refinement of detected hits. Compile-time constant (baked into the shader at construction).
 * @property {'linear'|'adaptive'} [tracing='linear'] - Adaptive mode uses a nearest-depth hierarchy and exact mip-zero candidates. Requires WebGPU.
 * @property {number} [maxCandidates=1] - Adaptive mode only: maximum candidate surfaces to validate.
 * @property {number} [glossyMinLevel=0] - Experimental, adaptive stochastic mode only: rays from surfaces rougher than 0.1 stop descending at this Hi-Z level (FidelityFX's most_detailed_mip). `0` disables it. Use with `maxCandidates > 1`.
 * @property {boolean} [occlusionSkipping=false] - Adaptive mode only: lets rays pass behind occluders by skipping Hi-Z cells they are entirely hidden behind, as linear tracing does sample by sample. Recovers reflections of surfaces seen past an occluder at the cost of longer rays.
 * @property {?HiZNode} [hiZ=null] - Optional shared hierarchy built from the same depth texture.
 * @property {Camera} [camera=null] - Camera the scene is rendered with. Inferred from the color pass when omitted.
 */

/**
 * Post processing node for computing screen space reflections (SSR).
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
	 * @param {Node<vec4>} colorNode - The node that represents the beauty pass.
	 * @param {Node<float>} depthNode - A node that represents the beauty pass's depth.
	 * @param {Node<vec3>} normalNode - A node that represents the beauty pass's normals.
	 * @param {SSRNodeOptions} [options] - Optional inputs for material and environment data.
	 */
	constructor( colorNode, depthNode, normalNode, options = {} ) {

		super( 'vec4' );

		const {
			stochastic = false,
			metalnessNode = null,
			specularNode = null,
			specularF90Node = null,
			roughnessNode = null,
			reflectNonMetals = false,
			environmentNode = null,
			envImportanceSampling = false,
			diffuseNode = null,
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
		 * When `true`, env-luminance CDF tables are built and MIS is used for environment misses.
		 * Fixed at construction time.
		 *
		 * @type {boolean}
		 */
		this.envImportanceSampling = envImportanceSampling;

		/**
		 * The node that represents the beauty pass.
		 *
		 * @type {Node<vec4>}
		 */
		this.colorNode = colorNode;

		/**
		 * A node that represents the scene's diffuse color (typically the MRT `diffuseColor` attachment).
		 * When `null`, the shader uses `vec3(1)`.
		 *
		 * @type {?Node<vec4>}
		 */
		this.diffuseNode = diffuseNode !== null ? nodeObject( diffuseNode ) : null;

		/**
		 * A node that represents the beauty pass's depth.
		 *
		 * @type {Node<float>}
		 */
		this.depthNode = depthNode;
		this._tracing = options.tracing ?? 'linear';
		if ( ! [ 'linear', 'adaptive' ].includes( this._tracing ) ) throw new Error( 'SSRNode: Invalid tracing mode.' );
		this._maxCandidates = options.maxCandidates ?? 1;
		if ( ! Number.isInteger( this._maxCandidates ) || this._maxCandidates < 1 || this._maxCandidates > 4096 ) throw new Error( 'SSRNode: maxCandidates must be an integer from 1 to 4096.' );
		this._glossyMinLevel = options.glossyMinLevel ?? 0;
		this._occlusionSkipping = options.occlusionSkipping ?? false;
		this._showIterations = false;
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
		 * Stochastic mode only: per-pixel normal-incidence reflectance (F0), typically an MRT
		 * `specularColorBlended` attachment, which already includes IOR, specular intensity and
		 * metalness (raw `material.specularColor` reflects too brightly). When `null`, F0 is
		 * derived from metalness and the diffuse color.
		 *
		 * @type {?Node<vec3>}
		 * @default null
		 */
		this.specularNode = specularNode;

		/**
		 * Stochastic mode only: per-pixel grazing reflectance (F90), typically an MRT
		 * `specularF90` attachment. When `null`, the shader uses `1`.
		 *
		 * @type {?Node<float>}
		 * @default null
		 */
		this.specularF90Node = specularF90Node;

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
		 * The resolution scale. Valid values are in the range
		 * `[0,1]`. `1` means best quality but also results in
		 * more computational overhead. Setting to `0.5` means
		 * the effect is computed in half-resolution.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.resolutionScale = 1;

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
		 * A multiplier for the overall reflection intensity. `1` leaves the
		 * reflections unchanged, lower values dim them and higher values boost them.
		 *
		 * @type {UniformNode<float>}
		 * @default 1
		 */
		this.intensity = uniform( 1 );

		/**
		 * Screen-edge fade width, in UV units. As a screen-space hit approaches a screen
		 * border, the reflection is faded over this distance — either toward the environment
		 * reflection ({@link SSRNode#screenEdgeFadeBlack} `false`) or to zero intensity
		 * (`true`). `0` disables it.
		 *
		 * @type {UniformNode<float>}
		 * @default 0.2
		 */
		this.screenEdgeFade = uniform( 0.2 );

		/**
		 * When `true`, SSR fades to zero near screen borders instead of blending toward
		 * the environment map. Hits are faded by the reflection sample UV; misses are
		 * faded by the surface pixel UV.
		 *
		 * Baked into the shader as a compile-time constant so the unused fade branch is
		 * eliminated; assigning a new value recompiles the SSR material.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this._screenEdgeFadeBlack = false;

		/**
		 * Absolute env luminance cap. HDR env samples above this are scaled down (hue preserved).
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

		/**
		 * HDR environment map for screen-space misses.
		 *
		 * @type {?Texture}
		 */
		this.environmentNode = environmentNode;

		/**
		 * A node that represents the history texture for multi-bounce reflections.
		 *
		 * @type {?Texture}
		 */
		this.historyTexture = null;

		/**
		 * A node that represents the velocity texture for reprojection.
		 *
		 * @type {?Node<vec2>}
		 */
		this.velocityTexture = null;

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
		 * CDF-backed environment sampler. Created when {@link setEnvMap} is called.
		 *
		 * @private
		 * @type {?ImportanceSampledEnvironment}
		 */
		this._importanceEnvironment = null;

		/**
		 * Intensity multiplier applied to environment-map reflections on screen-space
		 * misses and at screen edges. Defaults to π to match the former hardcoded multiplier.
		 *
		 * @type {UniformNode<float>}
		 * @default Math.PI
		 */
		this.environmentIntensity = uniform( Math.PI );

		/**
		 * The render target the SSR is rendered into.
		 *
		 * @private
		 * @type {RenderTarget}
		 */
		this._ssrRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
		this._ssrRenderTarget.texture.name = 'SSRNode.SSR';

		/**
		 * The render target for the blurred SSR reflections.
		 *
		 * @private
		 * @type {RenderTarget}
		 */
		this._blurRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType, minFilter: LinearMipmapLinearFilter, magFilter: LinearFilter } );
		this._blurRenderTarget.texture.name = 'SSRNode.Blur';
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
		this._textureNode = passTexture( this, this._ssrRenderTarget.texture );

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

		if ( environmentNode !== null && environmentNode.isTexture === true ) {

			this.setEnvMap( environmentNode );

		}

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

	/** Candidate-surface budget. Changing it recompiles adaptive tracing. @type {number} */
	get maxCandidates() {

		return this._maxCandidates;

	}

	set maxCandidates( value ) {

		if ( ! Number.isInteger( value ) || value < 1 || value > 4096 ) throw new Error( 'SSRNode: maxCandidates must be an integer from 1 to 4096.' );
		if ( value === this._maxCandidates ) return;
		this._maxCandidates = value;
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

	/** Coarsest-leaf level for rough stochastic rays; `0` is off. Changing it recompiles adaptive tracing. @type {number} */
	get glossyMinLevel() {

		return this._glossyMinLevel;

	}

	set glossyMinLevel( value ) {

		if ( value === this._glossyMinLevel ) return;
		this._glossyMinLevel = value;
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
	 * Whether SSR fades to black near screen borders (compile-time constant). Assigning
	 * a new value recompiles the SSR material.
	 *
	 * @type {boolean}
	 */
	get screenEdgeFadeBlack() {

		return this._screenEdgeFadeBlack;

	}

	set screenEdgeFadeBlack( value ) {

		if ( value !== this._screenEdgeFadeBlack ) {

			this._screenEdgeFadeBlack = value;
			this._buildSSRMaterial();

		}

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
	 * {@link SSRNode#stepExponent}, {@link SSRNode#screenEdgeFadeBlack}) at their current
	 * values. A no-op until {@link SSRNode#setup} has captured the `Fn`.
	 *
	 * @private
	 */
	_buildSSRMaterial() {

		if ( this._ssrFn === null ) return;

		this._ssrMaterial.contextNode = context( this._sharedContext );
		this._ssrMaterial.fragmentNode = this._ssrFn();
		this._ssrMaterial.needsUpdate = true;

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
	 * Sets the size of the effect.
	 *
	 * @param {number} width - The width of the effect.
	 * @param {number} height - The height of the effect.
	 */
	setSize( width, height ) {

		width = Math.round( this.resolutionScale * width );
		height = Math.round( this.resolutionScale * height );

		this._resolution.value.set( width, height );
		this._ssrRenderTarget.setSize( width, height );
		this._blurRenderTarget.setSize( width, height );

	}

	/**
	 * Wires the feedback inputs for multi-bounce reflections: the previous frame's
	 * denoised result (`history`) and the velocity buffer used to reproject it
	 * (`velocity`). `history` accepts the producing node (e.g. a
	 * {@link RecurrentDenoiseNode}) — its output render target is used — or a raw
	 * texture. Pass `null` for both to disable multi-bounce.
	 *
	 * @param {Texture} history
	 * @param {Node<vec2>} velocity
	 */
	setHistory( history, velocity ) {

		this.historyTexture = ( history && typeof history.getRenderTarget === 'function' )
			? history.getRenderTarget().texture
			: history;
		this.velocityTexture = velocity;

	}

	/**
	 * Sets the environment map for importance-sampled env lighting when
	 * screen-space rays miss. Call this whenever the scene's env map changes.
	 *
	 * Uses {@link ImportanceSampledEnvironment} (CDF + MIS adapted from
	 * [three-gpu-pathtracer](https://github.com/gkjohnson/three-gpu-pathtracer)).
	 *
	 * @param {Texture|null} hdr - The equirectangular HDR environment map, or null to disable.
	 * @see {@link https://github.com/gkjohnson/three-gpu-pathtracer}
	 */
	setEnvMap( hdr ) {

		if ( hdr === null ) {

			if ( this._importanceEnvironment !== null ) {

				this._importanceEnvironment.clear();
				this._importanceEnvironment = null;

			}

			this._buildSSRMaterial();
			return;

		}

		if ( hdr.image === undefined || hdr.image.data === undefined ) {

			console.warn( 'SSRNode: `environmentNode` / `setEnvMap()` expects an equirectangular HDR texture with CPU-side image data (e.g. RGBELoader). PMREM cubemaps and `scene.environment` are not supported.' );
			return;

		}

		if ( this._importanceEnvironment === null ) {

			this._importanceEnvironment = new ImportanceSampledEnvironment( this.envImportanceSampling );

		}

		this._importanceEnvironment.updateFrom( hdr );
		this._buildSSRMaterial();

	}

	/**
	 * Intensity multiplier for the importance-sampled env contribution.
	 * Only available after {@link setEnvMap} has been called.
	 *
	 * @type {?UniformNode<float>}
	 */
	get envMapIntensity() {

		return this._importanceEnvironment !== null ? this._importanceEnvironment.intensity : null;

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

		const size = renderer.getDrawingBufferSize( _size );

		_quadMesh.material = this._ssrMaterial;

		this.setSize( size.width, size.height );

		// Advance the noise index once per frame (matches SSGI / Denoise).
		this._noiseIndex.value = ( this._noiseIndex.value + 1 ) % 0x7fffffff;

		// clear

		renderer.setMRT( null );
		renderer.setClearColor( 0x000000, 0 );

		// ssr

		renderer.setRenderTarget( ssrRenderTarget );
		_quadMesh.name = 'SSR [ Reflections ]';
		_quadMesh.render( renderer );

		// blur (optional)

		if ( this.stochastic === false && this.roughnessNode !== null ) {

			// blur mips but leave the base mip unblurred

			for ( let i = 0; i < blurRenderTarget.texture.mipmaps.length; i ++ ) {

				_quadMesh.material = ( i === 0 ) ? this._copyMaterial : this._blurMaterial;

				this._blurSpread.value = i;
				renderer.setRenderTarget( blurRenderTarget, 0, i );
				_quadMesh.name = 'SSR [ Blur Level ' + i + ' ]';
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

		const ssr = Fn( () => {

			const adaptive = this.tracing === 'adaptive';
			if ( adaptive && builder.renderer.backend.isWebGPUBackend !== true ) throw new Error( 'SSRNode: Adaptive tracing requires WebGPU.' );

			const noise = this.stochastic === true ? sampleMarchNoise( uvNode, this._noiseIndex ) : null;
			// reconstruct at the depth texel center

			const depthSize = vec2( textureSize( this.depthNode ) ).toConst();
			const uvPos = uvNode.mul( depthSize ).floor().add( 0.5 ).div( depthSize ).toVar();

			const depth = sampleDepth( uvPos ).toVar();

			// Skip background pixels (cleared far-plane depth); the target is cleared each frame.
			isBackgroundDepth( depth ).discard();

			const viewPosition = getViewPosition( uvPos, depth, this._cameraProjectionMatrixInverse ).toVar();
			const worldPosition = this._cameraWorldMatrix.mul( vec4( viewPosition, 1.0 ) ).xyz.toVar();
			const viewNormal = this.normalNode.rgb.normalize().toVar();

			const viewIncidentDir = ( ( this.camera.isPerspectiveCamera ) ? normalize( viewPosition ) : vec3( 0, 0, - 1 ) ).toVar();

			// The node system samples the metalness/roughness textures at the current uv,
			// so no explicit sample() is needed here.
			const metalness = float( this.metalnessNode );

			if ( this.stochastic === false && this._reflectNonMetals === false ) {

				metalness.lessThanEqual( 0.0 ).discard();

			}

			const roughness = float( this.roughnessNode );
			const glossiness = min( roughness.div( 0.25 ), 1 ).oneMinus();
			// Only the fade-to-black miss path reads this, and that path is baked out otherwise.
			const surfaceBorderFactor = this.screenEdgeFadeBlack ? computeScreenBorderFactor( uvPos, this.screenEdgeFade ) : null;
			const hitBorderWidth = this.screenEdgeFade.mul( glossiness );

			const V = viewIncidentDir.negate().normalize().toVar();

			let viewReflectDir, finalSampleWeight, specDominantFactor;
			let sampleEnvReflection = null;
			let samplePdf = null;

			if ( this.stochastic === false ) {

				viewReflectDir = reflect( viewIncidentDir, viewNormal ).normalize().toVar();
				finalSampleWeight = vec3( metalness );
				specDominantFactor = float( 1 );

			} else {

				const Xi = noise.toVar();
				// mirror bias trades the unbiased GGX lobe for lower variance
				Xi.y.assign( mix( Xi.y, 0.0, this.mirrorBias.mul( Xi.w.sqrt() ) ) );

				// The base color only feeds the metalness F0 fallback.
				const albedo = ( this.specularNode === null && this.diffuseNode !== null ? this.diffuseNode.sample( uvPos ).rgb : vec3( 1 ) ).toVar();
				const ggxSample = ggxReflectionSample( viewNormal, V, roughness, metalness, albedo, Xi, this.specularNode, this.specularF90Node ?? 1 ).toVar();

				// Sometimes the GGX sample is facing away from the surface, so we need to re-sample.
				If( ggxSample.get( 'reflectDir' ).dot( viewNormal ).lessThan( 0 ), () => {

					ggxSample.assign( ggxReflectionSample( viewNormal, V, roughness, metalness, albedo, Xi.add( Xi.mul( 7 ) ).fract(), this.specularNode, this.specularF90Node ?? 1 ) );

				} );

				viewReflectDir = ggxSample.get( 'reflectDir' ).toVar();
				samplePdf = ggxSample.get( 'pdf' );
				finalSampleWeight = ggxSample.get( 'sampleWeight' ).toVar();
				specDominantFactor = getSpecularDominantFactor( ggxSample.get( 'NdotV' ), roughness ).toVar();

				sampleEnvReflection = () => {

					if ( this._importanceEnvironment === null ) return vec3( 0 );

					const envColor = vec3( 0 ).toVar();

					if ( this.envImportanceSampling ) {

						const Xi2 = bindAnalyticNoise( this._resolution, 59 )( uvPos, this._noiseIndex );

						envColor.assign( this._importanceEnvironment.sampleEnvironmentMIS( {
							cameraWorldMatrix: this._cameraWorldMatrix,
							viewReflectDir,
							N: viewNormal,
							V,
							alpha: ggxSample.get( 'alpha' ),
							f0: ggxSample.get( 'f0' ),
							f90: ggxSample.get( 'f90' ),
							Xi2
						} ) );

					} else {

						envColor.assign( this._importanceEnvironment.sampleEnvironmentBRDF( {
							cameraWorldMatrix: this._cameraWorldMatrix,
							viewReflectDir,
							N: viewNormal,
							V,
							alpha: ggxSample.get( 'alpha' ),
							f0: ggxSample.get( 'f0' ),
							f90: ggxSample.get( 'f90' )
						} ) );

					}

					return envColor;

				};

			}

			// Multi-bounce: fold in the previous frame's reflection at the hit point, reprojected by its
			// own motion. The (1 - history.a) decay damps the feedback. No-op until both textures are set.
			const reprojectHitPointHistory = ( uvHit, color ) => {

				if ( ! ( this.historyTexture && this.velocityTexture ) ) return color;

				const velocity = this.velocityTexture.sample( uvHit ).xy;
				const historyUV = uvHit.sub( velocity );
				const historyBounce = texture( this.historyTexture, historyUV ).toVar();
				const sampleDecay = historyBounce.a.oneMinus();

				return color.add( historyBounce.rgb.mul( sampleDecay ) );

			};

			// Fades a screen-space hit near the screen borders, using the hit sample UV (where the
			// screen-space data was read). `screenEdgeFadeBlack` is baked, so the two modes branch in
			// JS: fade the reflection to black, or blend it toward the environment reflection.
			const applyHitEdgeFade = ( reflectColor, uvS, hitBorderWidth ) => {

				if ( this.screenEdgeFadeBlack ) {

					const hitBorderFactor = computeScreenBorderFactor( uvS, this.screenEdgeFade );
					reflectColor.rgb.mulAssign( hitBorderFactor );

				} else {

					const hitBorderFactor = computeScreenBorderFactor( uvS, hitBorderWidth );

					If( hitBorderFactor.lessThan( 1 ), () => {

						reflectColor.rgb.assign( mix( sampleEnvReflection().mul( this.environmentIntensity ), reflectColor.rgb, hitBorderFactor ) );

					} );

				}

			};

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

			const d0 = uvPos.mul( this._resolution ).xy.toVar();
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
			const hitUvS = vec2( 0 ).toVar();
			const hitD = float( 0 ).toVar();
			const validatedPosition = vec3( 0 ).toVar();
			const adaptiveRay = d1viewPosition.sub( viewPosition ).toConst();
			const surfacePoint = ( uv, raw ) => getViewPosition( uv, projectionDepth( raw, this._cameraNear, this._cameraFar ), this._cameraProjectionMatrixInverse );
			const nearRay = ( point ) => {

				const pixelWidth = getSSRPixelWidth( this._cameraProjectionMatrixInverse, point.z, invResolution.x, this.camera.isPerspectiveCamera );
				return validateSSRRayDistance( point, viewPosition, adaptiveRay, max( pixelWidth.mul( 3 ), this.thickness ) );

			};

			const iterationHeat = ( fraction, exhausted ) => {

				const cool = mix( vec3( 0, 0, 1 ), vec3( 0, 1, 0 ), fraction.mul( 2 ).clamp() );
				const warm = mix( cool, vec3( 1, 0, 0 ), fraction.mul( 2 ).sub( 1 ).clamp() );
				return vec4( bool( exhausted ).select( vec3( 1 ), warm ), 1 );

			};

			const validateCandidate = ( hitUV, raw ) => {

				const valid = bool( false ).toVar();
				const allowed = dot( viewReflectDir, this.normalNode.sample( hitUV ).rgb ).lessThan( 0 );
				If( allowed, () => {

					const point = surfacePoint( hitUV, raw ).toVar();
					validatedPosition.assign( point );
					valid.assign( nearRay( point ) );

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
					vec3( uvPos, toDepth( viewPosition.z, this._cameraNear, this._cameraFar ) ),
					vec3( d1.mul( invResolution ), toDepth( d1viewPosition.z, this._cameraNear, this._cameraFar ) ), {
						reversedDepth: reversed,
						start: rayLen.reciprocal(),
						maxSteps: this.adaptiveMaxSteps,
						maxCandidates: this.maxCandidates,
						minLevel: this.stochastic && this.glossyMinLevel > 0 ? roughness.greaterThan( 0.1 ).select( int( this.glossyMinLevel ), int( 0 ) ) : 0,
						depthTolerance: 2e-7,
						projectDepth: raw => projectionDepth( raw, this._cameraNear, this._cameraFar ),
						// The cell's farthest surface fails the thickness test where the ray
						// enters and leaves, so every nearer surface in between fails it too.
						isHidden: this.occlusionSkipping === false ? null : ( uvIn, uvOut, raw ) => nearRay( surfacePoint( uvIn, raw ).toVar() ).not().and( nearRay( surfacePoint( uvOut, raw ).toVar() ).not() ),
						acceptHit: ( uv, s, raw ) => validateCandidate( uv, raw )
					} );
				if ( this._traceDebug === true ) return vec4( float( result.iterations ), float( result.candidates ), float( result.hit ), float( result.exhausted ) );
				if ( this.showIterations ) return iterationHeat( float( result.iterations ).div( float( this.adaptiveMaxSteps ) ), result.exhausted );
				foundHit.assign( result.hit );
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

					withinRange.assign( withinRange.and( dot( viewReflectDir, this.normalNode.sample( uvS ).rgb.normalize() ).lessThan( 0 ) ) );

				}

				If( withinRange, () => {

					const hitWorldPosition = this._cameraWorldMatrix.mul( vec4( vP, 1.0 ) ).xyz.toVar();
					const worldDistance = distance( worldPosition, hitWorldPosition ).mul( specDominantFactor ).toVar();

					let reflectSample = this.colorNode.sample( uvS );

					if ( this.stochastic === true ) {

						// Cone footprint (filtered importance sampling): the ray stands for
						// 1 / ( coneSamples * pdf ) steradians, a disc of that solid angle at the hit.
						// ponytail: isotropic footprint, ignores the hit surface's tilt and depth edges;
						// a depth-aware Gaussian pyramid is the upgrade if leaking shows.
						const radius = distance( viewPosition, vP ).mul( samplePdf.mul( this.coneSamples ).mul( Math.PI ).max( 1e-6 ).inverseSqrt() );
						const pixelSize = getViewPosition( uvS.add( uvPixelStepX ), hitD, this._cameraProjectionMatrixInverse ).x.sub( vP.x ).abs().max( 1e-6 );
						reflectSample = reflectSample.level( log2( radius.mul( 2 ).div( pixelSize ) ).clamp( 0, this.coneMaxLevel ) );

					}

					const reflectColor = reflectSample.toVar();

					// Multi-bounce: add the reprojected previous-frame reflection at the hit point.
					reflectColor.rgb.assign( reprojectHitPointHistory( uvS, reflectColor.rgb ) );

					if ( this.stochastic === true ) applyHitEdgeFade( reflectColor, uvS, hitBorderWidth );

					// The scatter (GGX) path bakes distance/grazing response into finalSampleWeight.
					// The mirror/blur path is a plain reflection, so reapply upstream's squared
					// distance attenuation and grazing Fresnel here to match its falloff.
					let weightedColor = reflectColor.rgb.mul( finalSampleWeight );

					if ( this.stochastic === false ) {

						const ratio = float( 1 ).sub( distancePointPlane.div( this.maxDistance ) ).toVar();
						const attenuation = ratio.mul( ratio ).toVar();
						const fresnelCoe = div( dot( viewIncidentDir, viewReflectDir ).add( 1 ), 2 ).toVar();
						weightedColor = weightedColor.mul( attenuation.mul( fresnelCoe ) );

					}

					hit.assign( 1 );
					output.assign( vec4( weightedColor, worldDistance ) );

				} );

			} );

			// Screen-space ray missed: environment fallback (MIS when CDF env is set up).
			If( hit.equal( 0 ), () => {

				if ( this.stochastic === true ) {

					output.assign( vec4( sampleEnvReflection().mul( this.environmentIntensity ), float( ENV_RAY_LENGTH ) ) );

					// Misses fade by the surface pixel UV (where the reflection is being shaded).
					if ( this.screenEdgeFadeBlack ) {

						output.rgb.mulAssign( surfaceBorderFactor );

					}

				}

			} );

			const lum = luminance( output.rgb ).max( 1e-4 ).toVar();
			output.rgb.mulAssign( this.maxLuminance.div( lum ).min( 1 ) );

			// scale the reflection color by the user-controlled intensity
			output.rgb.mulAssign( this.intensity );

			return output.max( 0 );

		} );

		this._ssrFn = ssr;
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
		this._blurRenderTarget.dispose();

		this._ssrMaterial.dispose();
		this._blurMaterial.dispose();
		this._copyMaterial.dispose();

		if ( this._importanceEnvironment !== null ) {

			this._importanceEnvironment.dispose();
			this._importanceEnvironment = null;

		}

	}

}

export default SSRNode;

/**
 * TSL function for creating screen space reflections (SSR).
 *
 * @tsl
 * @function
 * @param {Node<vec4>} colorNode - The node that represents the beauty pass.
 * @param {Node<float>} depthNode - A node that represents the beauty pass's depth.
 * @param {Node<vec3>} normalNode - A node that represents the beauty pass's normals.
 * @param {SSRNodeOptions} [options] - Optional inputs for material and environment data.
 * @returns {SSRNode}
 */
export const ssr = ( colorNode, depthNode, normalNode, options = {} ) => new SSRNode(
	nodeObject( colorNode ),
	nodeObject( depthNode ),
	nodeObject( normalNode ),
	options
);

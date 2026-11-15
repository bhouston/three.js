import { HalfFloatType, Vector2, RenderTarget, RendererUtils, QuadMesh, NodeMaterial, Node, NodeUpdateType, Matrix4, DepthTexture } from 'three/webgpu';
import { exp, float, Fn, max, texture, uniform, uv, vec2, vec4, luminance, convertToTexture, passTexture, velocity, ivec2, mix, If, struct, logarithmicDepthToViewZ, viewZToPerspectiveDepth, context, OnBeforeRenderPipeline, OnAfterRenderPipeline } from 'three/tsl';
import { clipAABB, computeHaltonOffsets, flickerReduction, sampleCurrentDepth, samplePreviousDepth } from '../utils/TAAUtils.js';

const _quadMesh = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();

let _rendererState;


/**
 * A special node that performs Temporal Anti-Aliasing Upscaling (TAAU).
 *
 * Like TRAA, the node accumulates jittered samples over multiple frames and
 * reprojects history with motion vectors. Unlike TRAA, the input buffers
 * (beauty, depth, velocity) are expected to be rendered at a lower resolution
 * than the renderer's drawing buffer — typically by lowering the upstream
 * pass's resolution via {@link PassNode#setResolutionScale} — and the resolve
 * pass reconstructs an output-resolution image using a 9-tap Blackman-Harris
 * filter (Gaussian approximation) over the jittered input samples. The result
 * is an alternative to FSR2/3 that does anti-aliasing and upscaling in a
 * single pass.
 *
 * References:
 * - Karis, "High Quality Temporal Supersampling", SIGGRAPH 2014, {@link https://advances.realtimerendering.com/s2014/}
 * - Riley/Arcila, FidelityFX Super Resolution 2, GDC 2022, {@link https://gpuopen.com/download/GDC_FidelityFX_Super_Resolution_2_0.pdf}
 *
 * Experimental variant of TAAU. Uses ping-pong history and packs the persistent
 * thin-feature lock in alpha. The input/output must be opaque; RGB is the signal.
 * The original TAAU implementation is retained for comparison.
 *
 * Options select a five-tap cross neighborhood, sharper Catmull-Rom history,
 * reconstruction sharpness, and sample-distance weighted accumulation.
 * Note: MSAA must be disabled when TAAU is in use.
 *
 * @augments Node
 * @three_import import { taau } from 'three/addons/tsl/display/ResearchTAAUNode.js';
 */
class ResearchTAAUNode extends Node {

	static get type() {

		return 'ResearchTAAUNode';

	}

	/**
	 * Constructs a new TAAU node.
	 *
	 * @param {TextureNode} beautyNode - The texture node that represents the input of the effect.
	 * @param {TextureNode} depthNode - A node that represents the scene's depth.
	 * @param {TextureNode} velocityNode - A node that represents the scene's velocity.
	 * @param {Camera} camera - The camera the scene is rendered with.
	 */
	constructor( beautyNode, depthNode, velocityNode, camera, options = {} ) {

		super( 'vec4' );

		// Experimental opaque-color resolve: alpha stores the reprojected lock.
		this.options = { cross: false, historyFilter: 'bilinear', reconstructionSharpness: 2.29, sampleDistanceWeight: false, registerPreviousDepth: false, ...options };

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isResearchTAAUNode = true;

		/**
		 * The `updateBeforeType` is set to `NodeUpdateType.FRAME` since the node renders
		 * its effect once per frame in `updateBefore()`.
		 *
		 * @type {string}
		 * @default 'frame'
		 */
		this.updateBeforeType = NodeUpdateType.FRAME;

		/**
		 * The texture node that represents the input of the effect.
		 *
		 * @type {TextureNode}
		 */
		this.beautyNode = beautyNode;

		/**
		 * A node that represents the scene's depth.
		 *
		 * @type {TextureNode}
		 */
		this.depthNode = depthNode;

		/**
		 * A node that represents the scene's velocity.
		 *
		 * @type {TextureNode}
		 */
		this.velocityNode = velocityNode;

		/**
		 * The camera the scene is rendered with.
		 *
		 * @type {Camera}
		 */
		this.camera = camera;

		/**
		 * When the difference between the current and previous depth goes above this threshold,
		 * the history is considered invalid.
		 *
		 * @type {number}
		 * @default 0.0005
		 */
		this.depthThreshold = 0.0005;

		/**
		 * The depth difference within the 3×3 neighborhood to consider a pixel as an edge.
		 *
		 * @type {number}
		 * @default 0.001
		 */
		this.edgeDepthDiff = 0.001;

		/**
		 * The history becomes invalid as the pixel length of the velocity approaches this value.
		 *
		 * @type {number}
		 * @default 128
		 */
		this.maxVelocityLength = 128;

		/**
		 * Baseline weight applied to the current frame in the resolve. Lower
		 * values produce smoother results with longer accumulation but slower
		 * convergence on disoccluded regions; the motion factor is added on
		 * top, so fast-moving pixels still respond quickly.
		 *
		 * @type {number}
		 * @default 0.025
		 */
		this.currentFrameWeight = 0.025;

		/**
		 * The jitter index selects the current camera offset value.
		 *
		 * @private
		 * @type {number}
		 * @default 0
		 */
		this._jitterIndex = 0;

		/**
		 * A uniform node holding the current jitter offset in input-pixel
		 * units. The shader needs this to know where each input sample was
		 * actually rendered when computing per-tap reconstruction weights.
		 *
		 * @private
		 * @type {UniformNode<vec2>}
		 */
		this._jitterOffset = uniform( new Vector2() );
		this._jitterUV = new Vector2();
		this._previousJitterUV = uniform( new Vector2() );
		this._historyValid = uniform( false );
		this._needsRestart = true;

		/**
		 * The render target that represents the history of frame data.
		 * Sized to the renderer's drawing buffer (the output resolution).
		 *
		 * @private
		 * @type {?RenderTarget}
		 */
		this._historyRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
		this._historyRenderTarget.textures[ 0 ].name = 'ResearchTAAUNode.history.color';
		this._historyTextureNode = texture( this._historyRenderTarget.texture );

		/**
		 * The render target for the resolve. Sized to the renderer's drawing
		 * buffer (the output resolution).
		 *
		 * @private
		 * @type {?RenderTarget}
		 */
		this._resolveRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
		this._resolveRenderTarget.texture.name = 'ResearchTAAUNode.resolve';

		/**
		 * Render target whose depth attachment holds the previous frame's
		 * depth buffer. The depth texture must be owned by a render target
		 * so that `copyTextureToTexture` can copy into it on the WebGL
		 * backend, which uses a framebuffer blit and therefore needs the
		 * destination depth texture to be attached to a framebuffer. This
		 * render target is sized independently of the history target so it
		 * can match the (lower-resolution) input depth texture.
		 *
		 * @private
		 * @type {RenderTarget}
		 */
		this._previousDepthRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, depthTexture: new DepthTexture() } );
		this._previousDepthRenderTarget.depthTexture.name = 'ResearchTAAUNode.previousDepth';

		/**
		 * Material used for the resolve step.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._resolveMaterial = new NodeMaterial();
		this._resolveMaterial.name = 'TAAU.resolve';

		/**
		 * Material used to seed the history render target on resize. It
		 * performs a bilinear upscale of the current beauty buffer into the
		 * output-sized history target so that the first frames after a
		 * resize do not fade in from black.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._seedMaterial = new NodeMaterial();
		this._seedMaterial.name = 'TAAU.seed';

		/**
		 * The result of the effect is represented as a separate texture node.
		 *
		 * @private
		 * @type {PassTextureNode}
		 */
		this._textureNode = passTexture( this, this._resolveRenderTarget.texture );

		/**
		 * Used to save the original/unjittered projection matrix.
		 *
		 * @private
		 * @type {Matrix4}
		 */
		this._originalProjectionMatrix = new Matrix4();

		/**
		 * A uniform node holding the camera's near and far.
		 *
		 * @private
		 * @type {UniformNode<vec2>}
		 */
		this._cameraNearFar = uniform( new Vector2() );

		/**
		 * A uniform node holding the camera world matrix.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._cameraWorldMatrix = uniform( new Matrix4() );

		/**
		 * A uniform node holding the camera world matrix inverse.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._cameraWorldMatrixInverse = uniform( new Matrix4() );

		/**
		 * A uniform node holding the camera projection matrix inverse.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._cameraProjectionMatrixInverse = uniform( new Matrix4() );

		/**
		 * A uniform node holding the previous frame's view matrix.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._previousCameraWorldMatrix = uniform( new Matrix4() );

		/**
		 * A uniform node holding the previous frame's projection matrix inverse.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._previousCameraProjectionMatrixInverse = uniform( new Matrix4() );

		/**
		 * A texture node for the previous depth buffer.
		 *
		 * @private
		 * @type {TextureNode}
		 */
		this._previousDepthNode = texture( this._previousDepthRenderTarget.depthTexture );

	}

	/**
	 * Returns the result of the effect as a texture node.
	 *
	 * @return {PassTextureNode} A texture node that represents the result of the effect.
	 */
	getTextureNode() {

		return this._textureNode;

	}

	/**
	 * Discards accumulated history after a camera cut, input change or reactivation.
	 */
	reset() {

		this._historyValid.value = false;
		this._needsRestart = true;

	}

	/**
	 * Sets the output size of the effect (history and resolve targets). The
	 * previous-depth texture is sized independently in `updateBefore()` to
	 * track the scene's current depth texture.
	 *
	 * @param {number} outputWidth - The output width (drawing buffer width).
	 * @param {number} outputHeight - The output height (drawing buffer height).
	 */
	setSize( outputWidth, outputHeight ) {

		this._historyRenderTarget.setSize( outputWidth, outputHeight );
		this._resolveRenderTarget.setSize( outputWidth, outputHeight );

	}

	/**
	 * Defines the jitter in input-pixel units so that the sequence explores
	 * the entire low resolution pixel footprint.
	 *
	 * @param {number} inputWidth - The width of the input buffers the camera renders into.
	 * @param {number} inputHeight - The height of the input buffers the camera renders into.
	 */
	setViewOffset( inputWidth, inputHeight ) {

		// save original/unjittered projection matrix for velocity pass

		this.camera.updateProjectionMatrix();
		this._originalProjectionMatrix.copy( this.camera.projectionMatrix );

		velocity.setProjectionMatrix( this._originalProjectionMatrix );

		// The jitter covers one input pixel to recover output subpixel samples.

		const haltonOffset = _haltonOffsets[ this._jitterIndex ];
		const jitterX = ( haltonOffset[ 0 ] - 0.5 );
		const jitterY = ( haltonOffset[ 1 ] - 0.5 );

		// Texture UVs grow downward on the WebGPU backend, as does setViewOffset Y.
		// A positive view offset renders the sample at uv + jitterUV; registration
		// therefore samples the previous jittered depth at historyUV - jitterUV.
		this._previousJitterUV.value.copy( this._jitterUV );
		this._jitterUV.set( jitterX / inputWidth, jitterY / inputHeight );
		this._jitterOffset.value.set( jitterX, jitterY );

		this.camera.setViewOffset(

			inputWidth, inputHeight,

			jitterX, jitterY,

			inputWidth, inputHeight

		);

	}

	/**
	 * Clears the view offset from the scene's camera.
	 */
	clearViewOffset() {

		this.camera.clearViewOffset();

		velocity.setProjectionMatrix( null );

		// update jitter index

		this._jitterIndex ++;
		this._jitterIndex = this._jitterIndex % _haltonOffsets.length;

	}

	/**
	 * This method is used to render the effect once per frame.
	 *
	 * @param {NodeFrame} frame - The current node frame.
	 */
	updateBefore( frame ) {

		const { renderer } = frame;

		// store previous frame matrices before updating current ones

		this._previousCameraWorldMatrix.value.copy( this._cameraWorldMatrix.value );
		this._previousCameraProjectionMatrixInverse.value.copy( this._cameraProjectionMatrixInverse.value );

		// update camera matrices uniforms

		this._cameraNearFar.value.set( this.camera.near, this.camera.far );
		this._cameraWorldMatrix.value.copy( this.camera.matrixWorld );
		this._cameraWorldMatrixInverse.value.copy( this.camera.matrixWorldInverse );
		this._cameraProjectionMatrixInverse.value.copy( this.camera.projectionMatrixInverse );

		// the output dimensions are derived from the renderer's drawing buffer

		const drawingBufferSize = renderer.getDrawingBufferSize( _size );
		const outputWidth = drawingBufferSize.width;
		const outputHeight = drawingBufferSize.height;

		//

		_rendererState = RendererUtils.resetRendererState( renderer, _rendererState );

		//

		const needsRestart = this._needsRestart ||
			this._historyRenderTarget.width !== outputWidth ||
			this._historyRenderTarget.height !== outputHeight;

		this.setSize( outputWidth, outputHeight );

		// every time the dimensions change we need fresh history data

		if ( needsRestart === true ) {

			this._historyValid.value = false;

			// Seed the history with a bilinear upscale of the current beauty
			// buffer. Without this the first frames after a resize fade in
			// from black because the history target was cleared. The seed
			// material is a quad pass that samples beauty at output UVs, so
			// it produces an output-sized image regardless of the input size.

			renderer.setRenderTarget( this._historyRenderTarget );
			_quadMesh.material = this._seedMaterial;
			_quadMesh.name = 'TAAU.seed';
			_quadMesh.render( renderer );
			renderer.setRenderTarget( null );

		}

		// Both buffers alternate roles. This retains color and lock without a full-size copy.
		this._historyTextureNode.value = this._historyRenderTarget.texture;

		// resolve

		renderer.setRenderTarget( this._resolveRenderTarget );
		_quadMesh.material = this._resolveMaterial;
		_quadMesh.name = 'TAAU';
		_quadMesh.render( renderer );
		renderer.setRenderTarget( null );

		// update history

		this._textureNode.value = this._resolveRenderTarget.texture;
		const previous = this._historyRenderTarget;
		this._historyRenderTarget = this._resolveRenderTarget;
		this._resolveRenderTarget = previous;

		// The resolve above used the old validity value; expose the new history
		// to the next frame only after its output and lock have been written.
		this._historyValid.value = true;
		this._needsRestart = false;

		// Copy the current scene depth into the previous-depth texture. We
		// keep the destination size locked to the source's actual dimensions
		// so that any one-frame timing mismatch between the scene pass's depth
		// attachment and the beauty render target's bookkeeping cannot
		// produce a copy with mismatched extents (which WebGPU rejects for
		// depth/stencil formats).

		const currentDepth = this.depthNode.value;
		const srcW = currentDepth.image !== null && currentDepth.image !== undefined ? currentDepth.image.width : 0;
		const srcH = currentDepth.image !== null && currentDepth.image !== undefined ? currentDepth.image.height : 0;

		if ( srcW > 0 && srcH > 0 ) {

			if ( this._previousDepthRenderTarget.width !== srcW || this._previousDepthRenderTarget.height !== srcH ) {

				this._previousDepthRenderTarget.setSize( srcW, srcH );

			}

			const dstDepth = this._previousDepthRenderTarget.depthTexture;
			renderer.copyTextureToTexture( currentDepth, dstDepth );

		}

		// restore

		RendererUtils.restoreRendererState( renderer, _rendererState );

	}

	/**
	 * This method is used to setup the effect's render targets and TSL code.
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {PassTextureNode}
	 */
	setup( builder ) {

		this.depthNode.build( builder );
		this.velocityNode.build( builder );
		this.beautyNode.build( builder );

		if ( builder.renderPipeline && ! builder.context.renderPipelineState.viewOffsetOwner ) {

			builder.context.renderPipelineState.viewOffsetOwner = this;

			OnBeforeRenderPipeline( () => {

				const beautyRenderTarget = ( this.beautyNode.isRTTNode ) ? this.beautyNode.renderTarget : this.beautyNode.passNode.renderTarget;

				const inputWidth = beautyRenderTarget.texture.width;
				const inputHeight = beautyRenderTarget.texture.height;

				this.setViewOffset( inputWidth, inputHeight );

			} );

			OnAfterRenderPipeline( () => {

				this.clearViewOffset();

			} );

		}

		this._previousDepthRenderTarget.depthTexture.format = this.depthNode.value.format;
		this._previousDepthRenderTarget.depthTexture.type = this.depthNode.value.type;

		const historyNode = this._historyTextureNode;

		// --- TAAU resolve ---
		//
		// For each output pixel, we map its position into input-pixel space,
		// find the closest jittered input sample, and reconstruct the current
		// color as a weighted sum of the 3×3 neighborhood around that sample.
		// Each tap's weight is a Gaussian approximation of a Blackman-Harris
		// window evaluated at the distance between the tap's (jittered)
		// sample center and the output pixel center. The same neighborhood
		// also supplies the moments used for variance clipping of the
		// reprojected history, so no second neighborhood read is needed.


		const resolve = Fn( () => {

			const uvNode = uv();
			const inputSize = this.beautyNode.size(); // ivec2
			const inputSizeF = vec2( inputSize );

			// output pixel center in input-pixel coordinates

			const pIn = uvNode.mul( inputSizeF );

			// the input sample at integer texel (m, n) was rendered at world
			// position (m + 0.5 + jitter). Solving for the closest tap gives:

			const closestTapF = pIn.sub( vec2( 0.5 ).add( this._jitterOffset ) ).round();
			const closestTap = ivec2( closestTapF );

			// depth dilation around the closest input tap

			const currentDepth = ( this.options.cross ? sampleCrossDepth : sampleCurrentDepth )( this.depthNode, closestTapF, this._cameraNearFar );
			const closestDepth = currentDepth.get( 'closestDepth' );
			const closestPositionTexel = currentDepth.get( 'closestPositionTexel' );
			const farthestDepth = currentDepth.get( 'farthestDepth' );

			// reproject using the velocity sampled at the dilated depth tap

			const offsetUV = this.velocityNode.load( closestPositionTexel ).xy.mul( vec2( 0.5, - 0.5 ) );
			const historyUV = uvNode.sub( offsetUV );
			const previousDepthUV = this.options.registerPreviousDepth ? historyUV.sub( this._previousJitterUV ) : historyUV;
			const previousDepth = samplePreviousDepth( this._previousDepthNode, previousDepthUV, this._previousCameraProjectionMatrixInverse, this._previousCameraWorldMatrix, this._cameraWorldMatrixInverse, this._cameraNearFar, this.camera );

			// history validity

			const isValidUV = historyUV.greaterThanEqual( 0 ).all().and( historyUV.lessThanEqual( 1 ).all() );
			const isEdge = farthestDepth.sub( closestDepth ).greaterThan( this.edgeDepthDiff );
			const isDisocclusion = closestDepth.sub( previousDepth ).greaterThan( this.depthThreshold );
			const hasValidHistory = this._historyValid.and( isValidUV ).and( isEdge.or( isDisocclusion.not() ) );

			// 9-tap Blackman-Harris (Gaussian approximation) reconstruction
			// of the current frame color, plus moment accumulation for the
			// variance clip of the history.

			const sumColor = vec4( 0 ).toVar();
			const sumWeight = float( 0 ).toVar();
			const moment1 = vec4( 0 ).toVar();
			const moment2 = vec4( 0 ).toVar();

			const offsets = this.options.cross ? [[ 0, 0 ], [ - 1, 0 ], [ 1, 0 ], [ 0, - 1 ], [ 0, 1 ]] : [
				[ - 1, - 1 ], [ 0, - 1 ], [ 1, - 1 ],
				[ - 1, 0 ], [ 0, 0 ], [ 1, 0 ],
				[ - 1, 1 ], [ 0, 1 ], [ 1, 1 ]
			];

			for ( const [ x, y ] of offsets ) {

				const tap = closestTap.add( ivec2( x, y ) );
				const tapCenter = vec2( tap ).add( vec2( 0.5 ).add( this._jitterOffset ) );
				const delta = pIn.sub( tapCenter );
				const d2 = delta.dot( delta );
				const w = exp( d2.mul( - this.options.reconstructionSharpness ) );

				// Use max() to prevent NaN values from propagating.
				const c = this.beautyNode.load( tap ).max( 0 );

				sumColor.addAssign( c.mul( w ) );
				sumWeight.addAssign( w );

				moment1.addAssign( c );
				moment2.addAssign( c.pow2() );

			}

			const currentColor = sumColor.div( sumWeight.max( 1e-5 ) );

			// variance clipping using the moments we just gathered

			const N = float( offsets.length );
			const mean = moment1.div( N );
			const motionFactor = uvNode.sub( historyUV ).mul( inputSizeF ).length().div( this.maxVelocityLength ).saturate();
			const varianceGamma = mix( 0.5, 1, motionFactor.oneMinus().pow2() );
			const variance = moment2.div( N ).sub( mean.pow2() ).max( 0 ).sqrt().mul( varianceGamma );
			const minColor = mean.sub( variance );
			const maxColor = mean.add( variance );

			const historyColor = this.options.historyFilter === 'catmull-rom' ? sampleHistoryCatmullRom( historyNode, historyUV ) : historyNode.sample( historyUV );
			const clippedHistoryColor = clipAABB( mean.clamp( minColor, maxColor ), historyColor, minColor, maxColor );

			// Current weight. Under TAAU a single input frame covers less of
			// the output grid, so the baseline current weight is lower than
			// in standard TRAA to give the accumulator more frames to fill
			// in sub-pixel detail. Motion still biases toward the current
			// frame to keep disoccluded and fast-moving pixels responsive.

			const currentLuma = luminance( currentColor.rgb );
			const meanLuma = luminance( mean.rgb ).toConst();
			const thinFeature = currentLuma.sub( meanLuma ).abs().div( meanLuma.max( 1e-5 ) ).smoothstep( 0, 0.2 );

			// Gate the lock by a two-sided depth change check. The
			// existing `isDisocclusion` is one-sided (only fires when
			// the scene moves farther), but new geometry appearing
			// closer also makes the history stale.
			const isDepthChanged = closestDepth.sub( previousDepth ).abs().greaterThan( this.depthThreshold );
			const canLock = this._historyValid.and( isValidUV ).and( isDepthChanged.not() );
			const gatedThinFeature = canLock.select( thinFeature, float( 0 ) );

			const decay = isDisocclusion.select( 0, 0.5 );
			const lock = max( gatedThinFeature, historyNode.sample( historyUV ).a.mul( decay ) ).saturate();
			const lockedHistoryColor = mix( clippedHistoryColor, historyColor, lock );

			const closestDelta = pIn.sub( closestTapF.add( vec2( 0.5 ).add( this._jitterOffset ) ) );
			const sampleWeight = this.options.sampleDistanceWeight ? exp( closestDelta.dot( closestDelta ).mul( - 2 ) ).max( 0.1 ) : float( 1 );
			const currentWeight = float( this.currentFrameWeight ).mul( sampleWeight ).toVar();
			currentWeight.assign( hasValidHistory.select( currentWeight.add( motionFactor ).saturate(), 1 ) );

			const output = flickerReduction( currentColor, lockedHistoryColor, currentWeight );

			return vec4( output.rgb, lock );

		} );

		// materials

		const sharedContext = context( builder.getSharedContext() );

		this._resolveMaterial.contextNode = sharedContext;
		this._resolveMaterial.fragmentNode = resolve();

		this._seedMaterial.fragmentNode = vec4( this.beautyNode.sample( uv() ).rgb, 0 );

		this._seedMaterial.contextNode = sharedContext;

		return this._textureNode;

	}

	/**
	 * Frees internal resources. This method should be called
	 * when the effect is no longer required.
	 */
	dispose() {

		super.dispose();

		this._historyRenderTarget.dispose();
		this._resolveRenderTarget.dispose();
		this._previousDepthRenderTarget.dispose();

		this._resolveMaterial.dispose();
		this._seedMaterial.dispose();

	}

}

export default ResearchTAAUNode;

const _haltonOffsets = /*@__PURE__*/ computeHaltonOffsets( 32 );

/**
 * TSL function for creating a TAAU node for Temporal Anti-Aliasing Upscaling.
 *
 * @tsl
 * @function
 * @param {TextureNode} beautyNode - The texture node that represents the input of the effect.
 * @param {TextureNode} depthNode - A node that represents the scene's depth.
 * @param {TextureNode} velocityNode - A node that represents the scene's velocity.
 * @param {Camera} camera - The camera the scene is rendered with.
 * @returns {ResearchTAAUNode}
 */
export const researchTAAU = ( beautyNode, depthNode, velocityNode, camera, options = {} ) => new ResearchTAAUNode( convertToTexture( beautyNode ), depthNode, velocityNode, camera, options );

// Cross-shaped depth dilation drops the diagonal taps along with the color taps.
const crossDepthStruct = struct( { closestDepth: 'float', closestPositionTexel: 'vec2', farthestDepth: 'float' } );
const sampleCrossDepth = Fn( ( [ depthNode, positionTexel, cameraNearFar ], builder ) => {

	const closest = float( 2 ).toVar();
	const farthest = float( - 1 ).toVar();
	const position = vec2( positionTexel ).toVar();
	for ( const [ x, y ] of [[ 0, 0 ], [ - 1, 0 ], [ 1, 0 ], [ 0, - 1 ], [ 0, 1 ]] ) {

		const neighbor = positionTexel.add( vec2( x, y ) ).clamp( vec2( 0 ), vec2( depthNode.size() ).sub( 1 ) );
		let depth = depthNode.load( neighbor ).r;
		if ( builder.renderer.reversedDepthBuffer ) depth = depth.oneMinus();
		if ( builder.renderer.logarithmicDepthBuffer ) depth = viewZToPerspectiveDepth( logarithmicDepthToViewZ( depth, cameraNearFar.x, cameraNearFar.y ), cameraNearFar.x, cameraNearFar.y );
		depth = depth.toConst();
		If( depth.lessThan( closest ), () => {

			closest.assign( depth );
			position.assign( neighbor );

		} );
		farthest.assign( max( farthest, depth ) );

	}

	return crossDepthStruct( closest, position, farthest );

} );

// Five bilinear taps approximate Catmull-Rom by dropping its four small corner lobes.
// Renormalization preserves constants. Reference: Filmic SMAA, SIGGRAPH 2016.
const sampleHistoryCatmullRom = ( history, coordinate ) => {

	const size = vec2( history.size() ).toConst();
	const position = coordinate.mul( size ).sub( 0.5 ).toConst();
	const base = position.floor().toConst();
	const f = position.fract().toConst();
	const f2 = f.mul( f ).toConst();
	const f3 = f2.mul( f ).toConst();
	const w0 = f.mul( - 0.5 ).add( f2 ).sub( f3.mul( 0.5 ) ).toConst();
	const w1 = vec2( 1 ).sub( f2.mul( 2.5 ) ).add( f3.mul( 1.5 ) ).toConst();
	const w2 = f.mul( 0.5 ).add( f2.mul( 2 ) ).sub( f3.mul( 1.5 ) ).toConst();
	const w3 = f3.mul( 0.5 ).sub( f2.mul( 0.5 ) ).toConst();
	const w12 = w1.add( w2 ).toConst();
	const middle = base.add( 0.5 ).add( w2.div( w12 ) ).toConst();
	const left = base.sub( 0.5 ).toConst();
	const right = base.add( 2.5 ).toConst();
	const centerWeight = w12.x.mul( w12.y ).toConst();
	const weights = [ centerWeight, w0.x.mul( w12.y ), w3.x.mul( w12.y ), w12.x.mul( w0.y ), w12.x.mul( w3.y ) ];
	const positions = [ middle, vec2( left.x, middle.y ), vec2( right.x, middle.y ), vec2( middle.x, left.y ), vec2( middle.x, right.y ) ];
	let color = vec4( 0 );
	let total = float( 0 );
	for ( let i = 0; i < 5; i ++ ) {

		color = color.add( history.sample( positions[ i ].div( size ) ).mul( weights[ i ] ) );
		total = total.add( weights[ i ] );

	}

	return color.div( total ).max( 0 );

};

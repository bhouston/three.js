import { HalfFloatType, Vector2, RenderTarget, RendererUtils, QuadMesh, NodeMaterial, Node, NodeUpdateType, Matrix4 } from 'three/webgpu';
import { float, Fn, max, texture, uniform, uv, vec2, vec4, convertToTexture, passTexture, velocity, ivec2, mix, smoothstep, context, OnBeforeRenderPipeline, OnAfterRenderPipeline } from 'three/tsl';
import { clipAABB, computeHaltonOffsets, flickerReduction, sampleCatmullRom, sampleCurrentDepth } from '../utils/TAAUtils.js';
import { Reprojection } from '../../utils/Reprojection.js';

const _quadMesh = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();
const _jitterMatrix = /*@__PURE__*/ new Matrix4();

let _rendererState;


/**
 * A special node that applies TRAA (Temporal Reprojection Anti-Aliasing).
 *
 * References:
 * - {@link https://alextardif.com/TAA.html}
 * - {@link https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/}
 *
 * Note: MSAA must be disabled when TRAA is in use.
 *
 * @augments Node
 * @three_import import { traa } from 'three/addons/tsl/display/TRAANode.js';
 */
class TRAANode extends Node {

	static get type() {

		return 'TRAANode';

	}

	/**
	 * Constructs a new TRAA node.
	 *
	 * @param {TextureNode} beautyNode - The texture node that represents the input of the effect.
	 * @param {TextureNode} depthNode - A node that represents the scene's depth.
	 * @param {TextureNode} velocityNode - A node that represents the scene's velocity, with `viewZDelta` in the alpha channel.
	 * @param {Camera} camera - The camera the scene is rendered with.
	 * @param {?TextureNode} [previousDepthNode=null] - The previous frame's depth, e.g. `pass.getPreviousTextureNode( 'depth' )`. By default the depth is copied each frame.
	 */
	constructor( beautyNode, depthNode, velocityNode, camera, previousDepthNode = null ) {

		super( 'vec4' );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isTRAANode = true;

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
		 * A node that represents the scene's velocity.
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
		 * The history is rejected when the previous frame's surface is nearer than the
		 * expected one by more than this fraction of its distance.
		 *
		 * @type {number}
		 * @default 0.002
		 */
		this.depthThreshold = 0.002;

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
		 * How the reprojected history is sampled. `'linear'` uses a single bilinear fetch which is cheaper
		 * but blurs the image under motion, since the history is resampled every frame. `'catmull-rom'`
		 * uses a 5-tap Catmull-Rom filter that keeps the history sharp.
		 *
		 * @type {('linear'|'catmull-rom')}
		 * @default 'catmull-rom'
		 */
		this.historySampling = 'catmull-rom';

		/**
		 * Whether to decrease the weight on the current frame when the velocity is more subpixel.
		 * This reduces blurriness under motion, but can introduce a square pattern artifact.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.useSubpixelCorrection = true;

		/**
		 * Whether the history is clipped to its neighborhood while the camera is at rest. Disabling it
		 * relaxes the clip over `historyLength` frames once the camera stops, which removes the shimmer
		 * of a still image, and restores it as soon as the camera moves. Changes without velocity
		 * (lighting, reflections) then lag behind; call `reset()` after one to restart the history.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.clipAtRest = true;

		/**
		 * How far the clip is relaxed, from `0` when the camera moved to `1` after it rested.
		 *
		 * @private
		 * @type {UniformNode<float>}
		 */
		this._restFactor = uniform( 0 );

		/**
		 * The number of frames the camera has been at rest.
		 *
		 * @private
		 * @type {number}
		 */
		this._restFrames = 0;

		/**
		 * The unjittered projection matrix of the previous frame.
		 *
		 * @private
		 * @type {Matrix4}
		 */
		this._previousProjectionMatrix = new Matrix4();

		/**
		 * Whether the history restarts from the next frame.
		 *
		 * @private
		 * @type {boolean}
		 */
		this._needsReset = false;

		/**
		 * The approximate number of frames the history accumulates. The current frame
		 * receives at least `1 / historyLength` of the blend weight, so `1` disables the history.
		 *
		 * @type {UniformNode<float>}
		 * @default 20
		 */
		this.historyLength = uniform( 20 );

		/**
		 * The jitter index selects the current camera offset value.
		 *
		 * @private
		 * @type {number}
		 * @default 0
		 */
		this._jitterIndex = 0;

		/**
		 * A uniform node holding the inverse resolution value.
		 *
		 * @private
		 * @type {UniformNode<vec2>}
		 */
		this._invSize = uniform( new Vector2() );

		/**
		 * The render target that represents the history of frame data.
		 *
		 * @private
		 * @type {?RenderTarget}
		 */
		this._historyRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
		this._historyRenderTarget.texture.name = 'TRAA / History';

		/**
		 * The render target for the resolve.
		 *
		 * @private
		 * @type {?RenderTarget}
		 */
		this._resolveRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
		this._resolveRenderTarget.texture.name = 'TRAA / Resolve';

		/**
		 * Material used for the resolve step.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._resolveMaterial = new NodeMaterial();
		this._resolveMaterial.name = 'TRAA / Resolve';

		/**
		 * The result of the effect is represented as a separate texture node.
		 *
		 * @private
		 * @type {PassTextureNode}
		 */
		this._textureNode = passTexture( this, this._resolveRenderTarget.texture );

		/**
		 * The render target of the debug output, created by `getDebugTextureNode()`.
		 *
		 * @private
		 * @type {?RenderTarget}
		 * @default null
		 */
		this._debugRenderTarget = null;

		/**
		 * The material that renders the debug output.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._debugMaterial = new NodeMaterial();
		this._debugMaterial.name = 'TRAA.debug';

		/**
		 * Used to save the original/unjittered projection matrix.
		 *
		 * @private
		 * @type {Matrix4}
		 */
		this._originalProjectionMatrix = new Matrix4();

		/**
		 * Whether the jitter is currently applied to the camera.
		 *
		 * @private
		 * @type {boolean}
		 * @default false
		 */
		this._viewOffsetEnabled = false;

		/**
		 * A uniform node holding the camera's near and far.
		 *
		 * @private
		 * @type {UniformNode<vec2>}
		 */
		this._cameraNearFar = uniform( new Vector2() );

		/**
		 * A texture node for the history.
		 *
		 * @private
		 * @type {TextureNode}
		 */
		this._historyTextureNode = texture( this._historyRenderTarget.texture );

		/**
		 * Maps the current frame to the previous one.
		 *
		 * @private
		 * @type {Reprojection}
		 */
		this._reprojection = new Reprojection( depthNode, camera, previousDepthNode );

		if ( previousDepthNode === null ) this._reprojection.previousDepthNode.value.name = 'TRAA / Previous Depth';

		/**
		 * The node used to render the scene's velocity.
		 *
		 * @private
		 * @type {?VelocityNode}
		 */
		this._velocityNode = null;

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
	 * Returns a texture node that shows what the resolve did with the history, for debugging.
	 * Red is `1` where the history was thrown away, as off the screen or disoccluded. Green is
	 * the weight of the current frame in the result. It is rendered with a second resolve from
	 * the first call on.
	 *
	 * @return {PassTextureNode} A texture node that represents the debug output.
	 */
	getDebugTextureNode() {

		if ( this._debugRenderTarget === null ) {

			this._debugRenderTarget = new RenderTarget( this._resolveRenderTarget.width, this._resolveRenderTarget.height, { depthBuffer: false } );
			this._debugRenderTarget.texture.name = 'TRAA / Debug';

		}

		return passTexture( this, this._debugRenderTarget.texture );

	}

	/**
	 * Sets the size of the effect.
	 *
	 * @param {number} width - The width of the effect.
	 * @param {number} height - The height of the effect.
	 */
	setSize( width, height ) {

		this._historyRenderTarget.setSize( width, height );
		this._resolveRenderTarget.setSize( width, height );
		if ( this._debugRenderTarget !== null ) this._debugRenderTarget.setSize( width, height );

		this._invSize.value.set( 1 / width, 1 / height );

	}

	/**
	 * Applies the TRAA's current jitter to the projection
	 * matrix of the scene's camera.
	 *
	 * @param {number} width - The width of the effect.
	 * @param {number} height - The height of the effect.
	 */
	setViewOffset( width, height ) {

		// save original/unjittered projection matrix for velocity pass

		this._originalProjectionMatrix.copy( this.camera.projectionMatrix );
		this._viewOffsetEnabled = true;

		this._velocityNode.setProjectionMatrix( this._originalProjectionMatrix );

		//

		const jitterOffset = _haltonOffsets[ this._jitterIndex ];

		this.camera.projectionMatrix.premultiply( _jitterMatrix.makeTranslation(

			- 2 * ( jitterOffset[ 0 ] - 0.5 ) / width,
			2 * ( jitterOffset[ 1 ] - 0.5 ) / height,
			0

		) );

		this.camera.projectionMatrixInverse.copy( this.camera.projectionMatrix ).invert();

	}

	/**
	 * Restores the original projection matrix of the scene's camera.
	 */
	clearViewOffset() {

		if ( this._viewOffsetEnabled === false ) return;

		this._viewOffsetEnabled = false;

		this.camera.projectionMatrix.copy( this._originalProjectionMatrix );
		this.camera.projectionMatrixInverse.copy( this._originalProjectionMatrix ).invert();

		this._velocityNode.setProjectionMatrix( null );

		// update jitter index

		this._jitterIndex ++;
		this._jitterIndex = this._jitterIndex % _haltonOffsets.length;

	}

	/**
	 * Restarts the history from the next frame, which then has the full weight.
	 */
	reset() {

		this._needsReset = true;

	}

	/**
	 * This method is used to render the effect once per frame.
	 *
	 * @param {NodeFrame} frame - The current node frame.
	 */
	updateBefore( frame ) {

		const { renderer } = frame;

		this._cameraNearFar.value.set( this.camera.near, this.camera.far );

		// keep the TRAA in sync with the dimensions of the beauty node

		const beautyRenderTarget = ( this.beautyNode.isRTTNode ) ? this.beautyNode.renderTarget : this.beautyNode.passNode.renderTarget;

		const width = beautyRenderTarget.texture.width;
		const height = beautyRenderTarget.texture.height;

		_rendererState = RendererUtils.resetRendererState( renderer, _rendererState );

		//

		const needsRestart = this._needsReset || this._reprojection.needsHistoryReset() ||
			this._historyRenderTarget.width !== width || this._historyRenderTarget.height !== height;
		this._needsReset = false;
		this.setSize( width, height );

		// the clip relaxes while the camera is at rest and is restored when it moves

		const atRest = needsRestart === false &&
			this.camera.matrixWorldInverse.equals( this._reprojection.previousCameraViewMatrix.value ) &&
			this._originalProjectionMatrix.equals( this._previousProjectionMatrix );

		this._previousProjectionMatrix.copy( this._originalProjectionMatrix );
		this._restFrames = atRest ? this._restFrames + 1 : 0;
		this._restFactor.value = Math.min( this._restFrames / this.historyLength.value, 1 );

		// swap history and resolve

		const historyRenderTarget = this._historyRenderTarget;
		this._historyRenderTarget = this._resolveRenderTarget;
		this._resolveRenderTarget = historyRenderTarget;

		this._historyTextureNode.value = this._historyRenderTarget.texture;
		this._textureNode.value = this._resolveRenderTarget.texture;

		// every time when the dimensions change we need fresh history data

		if ( needsRestart === true ) {

			// make sure to reset the history with the contents of the beauty buffer otherwise subsequent frames after the
			// resize will fade from a darker color to the correct one because the history was cleared with black.

			renderer.copyTextureToTexture( beautyRenderTarget.texture, this._historyRenderTarget.texture );

		}

		// resolve

		renderer.setRenderTarget( this._resolveRenderTarget );
		_quadMesh.material = this._resolveMaterial;
		_quadMesh.name = 'TRAA';
		_quadMesh.render( renderer );

		if ( this._debugRenderTarget !== null ) {

			renderer.setRenderTarget( this._debugRenderTarget );
			_quadMesh.material = this._debugMaterial;
			_quadMesh.name = 'TRAA / Debug';
			_quadMesh.render( renderer );

		}

		renderer.setRenderTarget( null );

		// store the depth and camera matrices for the next frame

		this._reprojection.update( renderer );

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

				const size = builder.renderer.getDrawingBufferSize( _size );
				this.setViewOffset( size.width, size.height );

			} );

			OnAfterRenderPipeline( () => {

				this.clearViewOffset();

			} );

		}

		if ( builder.context.velocity !== undefined ) {

			this._velocityNode = builder.context.velocity;

		} else {

			this._velocityNode = velocity;

		}

		// Performs variance clipping.
		// See: https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
		const varianceClipping = Fn( ( [ positionTexel, currentColor, historyColor, gamma ] ) => {

			const offsets = [
				[ - 1, - 1 ],
				[ - 1, 1 ],
				[ 1, - 1 ],
				[ 1, 1 ],
				[ 1, 0 ],
				[ 0, - 1 ],
				[ 0, 1 ],
				[ - 1, 0 ]
			];

			const moment1 = currentColor.toVar();
			const moment2 = currentColor.pow2().toVar();

			for ( const [ x, y ] of offsets ) {

				// Use max() to prevent NaN values from propagating.
				const neighbor = this.beautyNode.offset( ivec2( x, y ) ).load( positionTexel ).max( 0 );
				moment1.addAssign( neighbor );
				moment2.addAssign( neighbor.pow2() );

			}

			const N = float( offsets.length + 1 );
			const mean = moment1.div( N );
			const variance = moment2.div( N ).sub( mean.pow2() ).max( 0 ).sqrt().mul( gamma );
			const minColor = mean.sub( variance );
			const maxColor = mean.add( variance );

			return clipAABB( mean.clamp( minColor, maxColor ), historyColor, minColor, maxColor );

		} );

		// Returns the amount of subpixel (expressed within [0, 1]) in the velocity.
		const subpixelCorrection = Fn( ( [ velocityUV, textureSize ] ) => {

			const velocityTexel = velocityUV.mul( textureSize );
			const phase = velocityTexel.fract().abs();
			const weight = max( phase, phase.oneMinus() );
			return weight.x.mul( weight.y ).oneMinus().div( 0.75 );

		}, { velocityUV: 'vec2', textureSize: 'ivec2', return: 'float' } );

		const historyNode = this._historyTextureNode;

		const createResolve = ( debug ) => Fn( () => {

			const uvNode = uv();
			const textureSize = this.beautyNode.size(); // Assumes all the buffers share the same size.
			const positionTexel = uvNode.mul( textureSize );

			// sample the closest and farthest depths in the current buffer

			const currentDepth = sampleCurrentDepth( this.depthNode, positionTexel, this._cameraNearFar );
			const closestDepth = currentDepth.get( 'closestDepth' );
			const closestPositionTexel = currentDepth.get( 'closestPositionTexel' );
			const farthestDepth = currentDepth.get( 'farthestDepth' );

			const surfaceVelocity = this.velocityNode.load( closestPositionTexel );
			const historyUV = this._reprojection.getHistoryUV( uvNode, surfaceVelocity.xy ).toConst();

			// compare the expected and visible surface distances in the previous camera's view space

			const { x: near, y: far } = this._reprojection.previousCameraNearFar;
			const expectedDistance = this._reprojection.getExpectedPreviousViewZ( this.depthNode.load( closestPositionTexel ).r, surfaceVelocity.w, this._cameraNearFar ).negate().toConst();
			const sampledDistance = this._reprojection.getPreviousViewZ( historyUV ).negate();
			const isDisocclusion = expectedDistance.sub( sampledDistance ).greaterThan( expectedDistance.mul( this.depthThreshold ) );

			// a surface outside the previous camera's clip range has no history, even on edges

			const isValidHistoryPosition = historyUV.greaterThanEqual( 0 ).all().and( historyUV.lessThanEqual( 1 ).all() )
				.and( expectedDistance.greaterThanEqual( near.mul( float( 1 ).sub( this.depthThreshold ) ) ) )
				.and( expectedDistance.lessThanEqual( far.mul( float( 1 ).add( this.depthThreshold ) ) ) );

			const isEdge = farthestDepth.sub( closestDepth ).greaterThan( this.edgeDepthDiff );
			const hasValidHistory = isValidHistoryPosition.and( isEdge.or( isDisocclusion.not() ) );

			// sample the current and previous colors

			const currentColor = this.beautyNode.sample( uvNode );
			const historyColor = this.historySampling === 'catmull-rom' ? sampleCatmullRom( historyNode, historyUV, vec2( textureSize ) ) : historyNode.sample( historyUV );

			// increase the weight towards the current frame under motion

			const motionFactor = uvNode.sub( historyUV ).mul( textureSize ).length().div( this.maxVelocityLength ).saturate();
			const currentWeight = this.historyLength.reciprocal().toVar(); // A minimum weight

			if ( this.useSubpixelCorrection ) {

				// Increase the minimum weight towards the current frame when the velocity is more subpixel.
				currentWeight.addAssign( subpixelCorrection( uvNode.sub( historyUV ), textureSize ).mul( 0.25 ) );

			}

			currentWeight.assign( hasValidHistory.select( currentWeight.add( motionFactor ).saturate(), 1 ) );

			if ( debug ) return vec4( hasValidHistory.select( 0, 1 ), currentWeight, 0, 1 );

			// Perform neighborhood clipping/clamping. We use variance clipping here.

			// a 1σ box at rest clips converged sub-pixel detail, so longer histories lost accuracy
			const varianceGamma = mix( 0.5, 2, motionFactor.oneMinus().pow2() );
			let clippedHistoryColor = varianceClipping( positionTexel, currentColor, historyColor, varianceGamma );

			if ( this.clipAtRest === false ) {

				// only a pixel that does not move either keeps its history as it is

				const pixelMotion = uvNode.sub( historyUV ).mul( textureSize ).length();
				const relaxation = smoothstep( 0.01, 0.1, pixelMotion ).oneMinus().mul( this._restFactor );
				clippedHistoryColor = mix( clippedHistoryColor, historyColor, relaxation );

			}

			// flicker reduction based on luminance weighing

			const output = flickerReduction( currentColor, clippedHistoryColor, currentWeight );

			return vec4( output.rgb, 1 );

		} );

		// materials

		this._resolveMaterial.contextNode = context( builder.getSharedContext() );
		this._resolveMaterial.fragmentNode = createResolve( false )();

		this._debugMaterial.contextNode = this._resolveMaterial.contextNode;
		this._debugMaterial.fragmentNode = createResolve( true )();

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
		if ( this._debugRenderTarget !== null ) this._debugRenderTarget.dispose();

		this._reprojection.dispose();

		this._resolveMaterial.dispose();
		this._debugMaterial.dispose();

	}

}

export default TRAANode;

const _haltonOffsets = /*@__PURE__*/ computeHaltonOffsets( 32 );

/**
 * TSL function for creating a TRAA node for Temporal Reprojection Anti-Aliasing.
 *
 * @tsl
 * @function
 * @param {TextureNode} beautyNode - The texture node that represents the input of the effect.
 * @param {TextureNode} depthNode - A node that represents the scene's depth.
 * @param {TextureNode} velocityNode - A node that represents the scene's velocity, with `viewZDelta` in the alpha channel.
 * @param {Camera} camera - The camera the scene is rendered with.
 * @param {?TextureNode} [previousDepthNode=null] - The previous frame's depth, e.g. `pass.getPreviousTextureNode( 'depth' )`. By default the depth is copied each frame.
 * @returns {TRAANode}
 */
export const traa = ( beautyNode, depthNode, velocityNode, camera, previousDepthNode = null ) => new TRAANode( convertToTexture( beautyNode ), depthNode, velocityNode, camera, previousDepthNode );

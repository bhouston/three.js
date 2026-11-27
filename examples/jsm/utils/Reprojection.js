import { DepthTexture, RenderTarget, Vector2 } from 'three/webgpu';
import { depthToViewZ, texture, uniform, vec2 } from 'three/tsl';

const _createPreviousDepthRenderTarget = ( isDepthTexture ) => isDepthTexture
	? new RenderTarget( 1, 1, { depthBuffer: false, depthTexture: new DepthTexture(), count: 0 } )
	: new RenderTarget( 1, 1, { depthBuffer: false } );

/**
 * Maps the current frame to the previous one, for effects that reuse data of the
 * previous frame like temporal anti-aliasing, denoising or reflections.
 *
 * It holds the previous frame's depth and camera matrices for validating the history.
 *
 * Note: This class can only be used with `WebGPURenderer`.
 *
 * @private
 */
class Reprojection {

	/**
	 * Constructs a new reprojection.
	 *
	 * @param {TextureNode} depthNode - The scene's depth.
	 * @param {Camera} camera - The camera the scene is rendered with.
	 * @param {?TextureNode} [previousDepthNode=null] - The previous frame's depth. Copied in `update()` when omitted.
	 */
	constructor( depthNode, camera, previousDepthNode = null ) {

		/**
		 * The scene's depth.
		 *
		 * @type {TextureNode}
		 */
		this.depthNode = depthNode;

		/**
		 * The camera the scene is rendered with.
		 *
		 * @type {Camera}
		 */
		this.camera = camera;

		/**
		 * The render target that holds the depth of the previous frame, if not provided.
		 *
		 * @private
		 * @type {?RenderTarget}
		 */
		this._previousDepthRenderTarget = null;

		if ( previousDepthNode === null ) {

			this._previousDepthRenderTarget = _createPreviousDepthRenderTarget( true );
			this._previousDepthRenderTarget.depthTexture.name = 'Reprojection.previousDepth';

			previousDepthNode = texture( this._previousDepthRenderTarget.depthTexture );

		}

		/**
		 * The previous frame's depth.
		 *
		 * @type {TextureNode}
		 */
		this.previousDepthNode = previousDepthNode;

		/**
		 * The camera's world matrix.
		 *
		 * @type {UniformNode<mat4>}
		 */
		this.cameraWorldMatrix = uniform( camera.matrixWorld );

		/**
		 * The camera's view matrix.
		 *
		 * @type {UniformNode<mat4>}
		 */
		this.cameraViewMatrix = uniform( camera.matrixWorldInverse );

		/**
		 * The camera's projection matrix inverse.
		 *
		 * @type {UniformNode<mat4>}
		 */
		this.cameraProjectionMatrixInverse = uniform( camera.projectionMatrixInverse );

		/**
		 * The previous frame's camera view matrix.
		 *
		 * @type {UniformNode<mat4>}
		 */
		this.previousCameraViewMatrix = uniform( camera.matrixWorldInverse.clone() );

		/**
		 * The previous frame's camera projection matrix.
		 *
		 * @type {UniformNode<mat4>}
		 */
		this.previousCameraProjectionMatrix = uniform( camera.projectionMatrix.clone() );

		/**
		 * The previous frame's camera projection matrix inverse.
		 *
		 * @type {UniformNode<mat4>}
		 */
		this.previousCameraProjectionMatrixInverse = uniform( camera.projectionMatrixInverse.clone() );

		/**
		 * The previous frame's camera near and far, for decoding the previous frame's depth.
		 *
		 * @type {UniformNode<vec2>}
		 */
		this.previousCameraNearFar = uniform( new Vector2( camera.near, camera.far ) );

		/**
		 * The camera's view matrix when the camera was last stored.
		 *
		 * @private
		 * @type {Matrix4}
		 */
		this._cameraViewMatrix = camera.matrixWorldInverse.clone();

		/**
		 * The camera's projection matrix when the camera was last stored.
		 *
		 * @private
		 * @type {Matrix4}
		 */
		this._cameraProjectionMatrix = camera.projectionMatrix.clone();

		/**
		 * The camera's projection matrix inverse when the camera was last stored.
		 *
		 * @private
		 * @type {Matrix4}
		 */
		this._cameraProjectionMatrixInverse = camera.projectionMatrixInverse.clone();

		/**
		 * The camera's near and far when the camera was last stored.
		 *
		 * @private
		 * @type {Vector2}
		 */
		this._cameraNearFar = new Vector2( camera.near, camera.far );

		/**
		 * The depth's width when the camera was last stored.
		 *
		 * @private
		 * @type {number}
		 */
		this._width = 0;

		/**
		 * The depth's height when the camera was last stored.
		 *
		 * @private
		 * @type {number}
		 */
		this._height = 0;

	}

	/**
	 * Sets the scene's depth, e.g. to switch to a downsampled one. A downsampled depth
	 * is a color texture, so the previous depth is recreated to match. Call it before
	 * the effect is built.
	 *
	 * @param {TextureNode} depthNode - The scene's depth.
	 */
	setInputs( depthNode ) {

		this.depthNode = depthNode;

		// a provided previous depth already matches the depth

		if ( this._previousDepthRenderTarget === null ) return;

		const isDepthTexture = depthNode.value.isDepthTexture === true;

		if ( isDepthTexture !== ( this.previousDepthNode.value.isDepthTexture === true ) ) {

			const name = this.previousDepthNode.value.name;

			this._previousDepthRenderTarget.dispose();
			this._previousDepthRenderTarget = _createPreviousDepthRenderTarget( isDepthTexture );

			this.previousDepthNode.value = isDepthTexture ? this._previousDepthRenderTarget.depthTexture : this._previousDepthRenderTarget.texture;
			this.previousDepthNode.value.name = name;

		}

	}

	/**
	 * Returns the UV the surface at the given UV had in the previous frame.
	 *
	 * @param {Node<vec2>} uvNode - The UV in the current frame.
	 * @param {Node<vec2>} velocity - The surface velocity in NDC.
	 * @return {Node<vec2>} The UV in the previous frame.
	 */
	getHistoryUV( uvNode, velocity ) {

		// convert the NDC offset to UV offset

		return uvNode.sub( velocity.mul( vec2( 0.5, - 0.5 ) ) );

	}

	/**
	 * Returns the view z the surface had in the previous frame.
	 *
	 * @param {Node<float>} depth - The surface's depth in the current frame.
	 * @param {Node<float>} viewZDelta - The surface's `viewZDelta`.
	 * @param {Node<vec2>} cameraNearFar - The camera's near and far.
	 * @return {Node<float>} The expected view z in the previous frame.
	 */
	getExpectedPreviousViewZ( depth, viewZDelta, cameraNearFar ) {

		return depthToViewZ( depth, cameraNearFar.x, cameraNearFar.y, this.camera ).sub( viewZDelta );

	}

	/**
	 * Samples the view z of the previous frame at the given UV.
	 *
	 * @param {Node<vec2>} uvNode - The UV in the previous frame.
	 * @return {Node<float>} The visible view z in the previous frame.
	 */
	getPreviousViewZ( uvNode ) {

		const { x: near, y: far } = this.previousCameraNearFar;

		return depthToViewZ( this.previousDepthNode.sample( uvNode ).r, near, far, this.camera );

	}

	/**
	 * Returns whether the stored depth has a different size, format or type from
	 * the current depth. Check before rendering to restart incompatible history.
	 *
	 * @return {boolean} Whether the history needs to be reset.
	 */
	needsHistoryReset() {

		const depth = this.depthNode.value;
		const previousDepth = this.previousDepthNode.value;

		// a provided previous depth is resized together with the current one

		if ( this._previousDepthRenderTarget === null ) return this._width !== depth.image.width || this._height !== depth.image.height;

		return previousDepth.image.width !== depth.image.width ||
			previousDepth.image.height !== depth.image.height ||
			previousDepth.format !== depth.format ||
			previousDepth.type !== depth.type;

	}

	/**
	 * Stores the scene's depth and the camera matrices for the next frame. Call it
	 * once per frame after the effect has rendered.
	 *
	 * @param {Renderer} renderer - The renderer.
	 */
	update( renderer ) {

		const depth = this.depthNode.value;

		if ( this._previousDepthRenderTarget !== null ) {

			// a downsampled depth is a color texture, see setInputs()

			const previousDepth = this.previousDepthNode.value;

			if ( this.needsHistoryReset() ) {

				previousDepth.image.width = depth.image.width;
				previousDepth.image.height = depth.image.height;
				previousDepth.format = depth.format;
				previousDepth.type = depth.type;
				previousDepth.needsUpdate = true;

			}

			renderer.copyTextureToTexture( depth, previousDepth );

		}

		this._storeCamera();
		this.advance();

	}

	/**
	 * Advances the camera state by one frame. Call it before the history is read,
	 * instead of `update()`, when the previous depth is provided.
	 */
	advance() {

		this.previousCameraViewMatrix.value.copy( this._cameraViewMatrix );
		this.previousCameraProjectionMatrix.value.copy( this._cameraProjectionMatrix );
		this.previousCameraProjectionMatrixInverse.value.copy( this._cameraProjectionMatrixInverse );
		this.previousCameraNearFar.value.copy( this._cameraNearFar );

		this._storeCamera();

	}

	/**
	 * Stores the current camera state and depth size.
	 *
	 * @private
	 */
	_storeCamera() {

		const { camera } = this;
		const { width, height } = this.depthNode.value.image;

		this._cameraViewMatrix.copy( camera.matrixWorldInverse );
		this._cameraProjectionMatrix.copy( camera.projectionMatrix );
		this._cameraProjectionMatrixInverse.copy( camera.projectionMatrixInverse );
		this._cameraNearFar.set( camera.near, camera.far );
		this._width = width;
		this._height = height;

	}

	/**
	 * Frees internal resources. This method should be called
	 * when the reprojection is no longer required.
	 */
	dispose() {

		if ( this._previousDepthRenderTarget !== null ) this._previousDepthRenderTarget.dispose();

	}

}

export { Reprojection };

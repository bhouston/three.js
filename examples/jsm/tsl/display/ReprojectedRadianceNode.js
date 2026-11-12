import { Matrix4, Node, Vector2 } from 'three/webgpu';
import { Fn, float, getScreenPosition, getViewPosition, isBackgroundDepth, rtt, sample, textureSize, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';

/**
 * Provides the previous frame's beauty (lit scene color) reprojected into the
 * current frame, so screen-space effects like {@link SSRNode} and {@link SSGINode}
 * can read it through `.sample( uv )` in place of the current beauty.
 *
 * The pass keeps its previous color and depth by swapping textures (see
 * {@link PassNode#getPreviousTextureNode}), so the history is the pass output
 * of the previous frame without any effect composited on top.
 *
 * Three modes are supported:
 *
 * - `camera`: Reprojects per query with the previous camera matrices. Only correct for static geometry.
 * - `velocity`: Reprojects per query with the motion vector at the queried uv. Handles moving objects.
 * - `buffer`: Runs the velocity reprojection once per frame into a buffer that is
 * aligned with the current frame. Queries are a single texture fetch.
 *
 * The result holds radiance in rgb and validity in alpha. History that is missing,
 * off-screen or disoccluded yields zero radiance and zero alpha.
 *
 * @augments Node
 * @three_import import { reprojectedRadiance } from 'three/addons/tsl/display/ReprojectedRadianceNode.js';
 */
class ReprojectedRadianceNode extends Node {

	static get type() {

		return 'ReprojectedRadianceNode';

	}

	/**
	 * Constructs a new reprojected radiance node.
	 *
	 * @param {PassNode} passNode - The scene pass. The `velocity` and `buffer` modes require a `velocity` MRT output.
	 * @param {Object} [options] - The options.
	 * @param {('camera'|'velocity'|'buffer')} [options.mode='velocity'] - The reprojection mode.
	 * @param {number} [options.resolutionScale=0.5] - The resolution scale of the buffer. Only used by the `buffer` mode.
	 */
	constructor( passNode, { mode = 'velocity', resolutionScale = 0.5 } = {} ) {

		super( 'vec4' );

		/**
		 * The scene pass.
		 *
		 * @type {PassNode}
		 */
		this.passNode = passNode;

		/**
		 * The reprojection mode.
		 *
		 * @type {('camera'|'velocity'|'buffer')}
		 * @readonly
		 */
		this.mode = mode;

		const camera = passNode.camera;

		// The camera state the previous frame was rendered with. With TRAA, the projection includes the jitter.

		const capturedView = new Matrix4();
		const capturedProjection = new Matrix4();
		const capturedSize = new Vector2();
		const capturedJitter = new Vector2();
		const jitter = new Vector2();
		let capturedFrameId = - 1;

		this._previousProjection = uniform( new Matrix4() );
		this._previousProjectionInverse = uniform( new Matrix4() );
		this._jitterDelta = uniform( new Vector2() );
		this._valid = uniform( 0 );

		// Updated once per frame while the effect that samples this node renders. This happens
		// after the scene pass and before TRAA removes its jitter from the camera.

		this._previousView = uniform( new Matrix4() ).onFrameUpdate( ( { frameId } ) => {

			const { width, height } = passNode.renderTarget;
			const elements = camera.projectionMatrix.elements;

			// TRAA excludes the projection jitter from velocity, the beauty still contains it.

			jitter.set( - elements[ 8 ] * 0.5, elements[ 9 ] * 0.5 );

			// The history is unusable if it was not captured in the previous frame or has a different size.

			this._valid.value = ( frameId === capturedFrameId + 1 && capturedSize.x === width && capturedSize.y === height ) ? 1 : 0;
			this._previousView.value.copy( capturedView );
			this._previousProjection.value.copy( capturedProjection );
			this._previousProjectionInverse.value.copy( capturedProjection ).invert();
			this._jitterDelta.value.copy( capturedJitter ).sub( jitter );

			capturedView.copy( camera.matrixWorldInverse );
			capturedProjection.copy( camera.projectionMatrix );
			capturedSize.set( width, height );
			capturedJitter.copy( jitter );
			capturedFrameId = frameId;

		} );

		this._cameraWorldMatrix = uniform( camera.matrixWorld );
		this._cameraProjectionMatrixInverse = uniform( camera.projectionMatrixInverse );

		/**
		 * The buffer holding the reprojected history with premultiplied validity.
		 *
		 * @private
		 * @type {?RTTNode}
		 */
		this._buffer = null;

		if ( mode === 'buffer' ) {

			const premultiplied = sample( ( uv ) => Fn( () => {

				const history = this._reproject( uv ).toConst();

				return vec4( history.rgb.mul( history.a ), history.a );

			} )() );

			this._buffer = rtt( premultiplied, null, null, { resolutionScale, depthBuffer: false } );

		}

	}

	/**
	 * The resolution scale of the buffer. Only used by the `buffer` mode.
	 *
	 * @type {number}
	 */
	get resolutionScale() {

		return this._buffer !== null ? this._buffer.getResolutionScale() : 1;

	}

	set resolutionScale( value ) {

		if ( this._buffer !== null ) this._buffer.setResolutionScale( value );

	}

	/**
	 * Samples the reprojected history.
	 *
	 * @param {Node<vec2>} uv - The uv of the current frame.
	 * @return {Node<vec4>} The radiance in rgb and the validity in alpha.
	 */
	sample( uv ) {

		if ( this._buffer === null ) return this._reproject( uv );

		return Fn( () => {

			const history = this._buffer.sample( uv ).toConst();

			// Undo the premultiplication so partially valid texels do not darken the radiance.

			return history.a.greaterThan( 0.01 ).select( vec4( history.rgb.div( history.a ), history.a ), vec4( 0 ) );

		} )();

	}

	setup() {

		return this.sample( uv() );

	}

	/**
	 * Frees internal resources. This method should be called
	 * when the node is no longer required.
	 */
	dispose() {

		if ( this._buffer !== null ) this._buffer.dispose();

		super.dispose();

	}

	/**
	 * Reprojects the previous beauty to the given uv of the current frame.
	 *
	 * @private
	 * @param {Node<vec2>} uv - The uv of the current frame.
	 * @return {Node<vec4>} The radiance in rgb and the validity in alpha.
	 */
	_reproject( uv ) {

		return Fn( () => {

			const passNode = this.passNode;

			// The pass allocates its history when this is compiled for the first time.

			const previousColor = passNode.getPreviousTextureNode( 'output' );
			const previousDepth = passNode.getPreviousTextureNode( 'depth' );

			// Where the surface was in the previous frame if it did not move.

			const depth = passNode.getTextureNode( 'depth' ).sample( uv ).r.toConst();
			const viewPosition = getViewPosition( uv, depth, this._cameraProjectionMatrixInverse );
			const previousViewPosition = this._previousView.mul( this._cameraWorldMatrix.mul( vec4( viewPosition, 1 ) ) ).xyz.toConst();

			let historyUV = getScreenPosition( previousViewPosition, this._previousProjection );

			if ( this.mode !== 'camera' ) {

				// Velocity is the motion in NDC. The background has no velocity and keeps the camera reprojection.

				const velocity = passNode.getTextureNode( 'velocity' ).sample( uv ).xy;

				historyUV = isBackgroundDepth( depth ).select( historyUV, uv.sub( velocity.mul( vec2( 0.5, - 0.5 ) ) ).add( this._jitterDelta ) );

			}

			historyUV = historyUV.toConst();

			// The expected depth only accounts for camera motion. Object motion along the view
			// direction has to stay within the tolerance of one percent of the depth.

			const expectedZ = previousViewPosition.z;
			const size = vec2( textureSize( previousDepth ) ).toConst();
			const pixelDepth = this._previousProjectionInverse.element( 1 ).y.abs().mul( expectedZ.abs() ).mul( 3 ).div( size.y );
			const tolerance = pixelDepth.max( expectedZ.abs().mul( 0.01 ) ).toConst();

			const background = isBackgroundDepth( depth ).toConst();
			const inBounds = ( p ) => p.greaterThanEqual( 0 ).all().and( p.lessThanEqual( 1 ).all() );
			const valid = this._valid.greaterThan( 0 ).and( inBounds( historyUV ) ).and( expectedZ.lessThan( 0 ) ).toConst();

			const pixel = historyUV.mul( size ).sub( 0.5 ).toConst();
			const base = pixel.floor();
			const fraction = pixel.fract();

			const radiance = vec3( 0 ).toVar();
			const support = float( 0 ).toVar();

			// Validate each bilinear tap so the interpolation does not pull color across an occlusion edge.

			for ( let y = 0; y <= 1; y ++ ) {

				for ( let x = 0; x <= 1; x ++ ) {

					const tapUV = base.add( vec2( x, y ) ).add( 0.5 ).div( size ).toConst();
					const tapDepth = previousDepth.sample( tapUV ).r.toConst();
					const tapZ = getViewPosition( tapUV, tapDepth, this._previousProjectionInverse ).z;

					// The background only matches the background, surfaces have to match the expected depth.

					const matches = background.select( isBackgroundDepth( tapDepth ), tapZ.sub( expectedZ ).abs().lessThanEqual( tolerance ) );
					const accepted = valid.and( inBounds( tapUV ) ).and( matches );
					const weightX = x === 0 ? fraction.x.oneMinus() : fraction.x;
					const weightY = y === 0 ? fraction.y.oneMinus() : fraction.y;
					const weight = weightX.mul( weightY ).mul( float( accepted ) ).toConst();

					radiance.addAssign( previousColor.sample( tapUV ).rgb.mul( weight ) );
					support.addAssign( weight );

				}

			}

			// Rejected history contributes no radiance. It is never replaced with the color of another surface.

			return vec4( radiance.div( support.max( 1e-6 ) ), support );

		} )();

	}

}

export default ReprojectedRadianceNode;

/**
 * TSL function for creating a reprojected radiance node.
 *
 * @tsl
 * @function
 * @param {PassNode} passNode - The scene pass. The `velocity` and `buffer` modes require a `velocity` MRT output.
 * @param {Object} [options] - The options.
 * @param {('camera'|'velocity'|'buffer')} [options.mode='velocity'] - The reprojection mode.
 * @param {number} [options.resolutionScale=0.5] - The resolution scale of the buffer. Only used by the `buffer` mode.
 * @returns {ReprojectedRadianceNode}
 */
export const reprojectedRadiance = ( passNode, options ) => new ReprojectedRadianceNode( passNode, options );

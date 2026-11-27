import { RedFormat, RenderTarget, Vector2, RendererUtils, QuadMesh, Node, NodeMaterial, NodeUpdateType, UnsignedByteType } from 'three/webgpu';
import { reference, getScreenPosition, getViewPosition, float, Break, Loop, int, max, abs, If, interleavedGradientNoise, screenCoordinate, Fn, passTexture, uv, uniform, depthToViewZ, vec2, lightPosition, lightTargetPosition, fract, rand, mix, context, isBackgroundDepth, projectionDepth, getNormalFromDepth, vec3, viewZToPerspectiveDepth, viewZToReversedPerspectiveDepth, viewZToOrthographicDepth, viewZToReversedOrthographicDepth } from 'three/tsl';
import { hiZ } from './HiZNode.js';
import { traceHiZ } from '../utils/traceHiZ.js';

const _quadMesh = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();

const _spatialOffsets = [ 0, 0.5, 0.25, 0.75 ];

let _rendererState;

/**
 * Post processing node for applying Screen-Space Shadows (SSS) to a scene.
 *
 * Screen-Space Shadows (also known as Contact Shadows) should ideally be used to complement
 * traditional shadow maps. They are best suited for rendering detailed shadows of smaller
 * objects at a closer scale like intricate shadowing on highly detailed models. In other words:
 * Use Shadow Maps for the foundation and Screen-Space Shadows for the details.
 *
 * The shadows produced by this implementation might have too hard edges for certain use cases.
 * Use a box, gaussian or hash blur to soften the edges before doing the composite with the
 * beauty pass. Code example:
 *
 * ```js
 * const sssPass = sss( scenePassDepth, camera, mainLight );
 *
 * const sssBlur = boxBlur( sssPass.r, { size: 2, separation: 1 } ); // optional blur
 * ```
 *
 * Limitations:
 *
 * - Ideally the maximum shadow length should not exceed `1` meter. Otherwise the effect gets
 * computationally very expensive since more samples during the ray marching process are evaluated.
 * You can mitigate this issue by reducing the `quality` parameter.
 * - The effect can only be used with a single directional light, the main light of your scene.
 * This main light usually represents the sun or daylight.
 * - Like other Screen-Space techniques SSS can only honor objects in the shadowing computation that
 * are currently visible within the camera's view.
 *
 * References:
 * - {@link https://panoskarabelas.com/blog/posts/screen_space_shadows/}.
 * - {@link https://www.bendstudio.com/blog/inside-bend-screen-space-shadows/}.
 *
 * @augments Node
 * @three_import import { sss } from 'three/addons/tsl/display/SSSNode.js';
 */
class SSSNode extends Node {

	static get type() {

		return 'SSSNode';

	}

	/**
	 * Constructs a new SSS node.
	 *
	 * @param {TextureNode} depthNode - A texture node that represents the scene's depth.
	 * @param {Camera} camera - The camera the scene is rendered with.
	 * @param {DirectionalLight} mainLight - The main directional light of the scene.
	 */
	constructor( depthNode, camera, mainLight ) {

		super( 'float' );

		/**
		 * A node that represents the beauty pass's depth.
		 *
		 * @type {TextureNode}
		 */
		this.depthNode = depthNode;

		/**
		 * A node that represents the scene's normals in view space. The normals are reconstructed
		 * from the depth if no node is set.
		 *
		 * @type {?Node<vec3>}
		 * @default null
		 */
		this.normalNode = null;

		/**
		 * Maximum shadow length in world units. Longer shadows result in more computational
		 * overhead.
		 *
		 * @type {UniformNode<float>}
		 * @default 0.1
		 */
		this.maxDistance = uniform( 0.1, 'float' );

		/**
		 * Depth testing thickness.
		 *
		 * @type {UniformNode<float>}
		 * @default 0.01
		 */
		this.thickness = uniform( 0.01, 'float' );

		/**
		 * Shadow intensity. Must be in the range `[0, 1]`.
		 *
		 * @type {UniformNode<float>}
		 * @default 1.0
		 */
		this.shadowIntensity = uniform( 1.0, 'float' );

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
		 * @default 0.5
		 */
		this.quality = uniform( 0.5 );

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
		 * Whether to use temporal filtering or not. Setting this property to
		 * `true` requires the usage of `TRAANode`. This will help to reduce noise
		 * although it introduces typical TAA artifacts like ghosting and temporal
		 * instabilities.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.useTemporalFiltering = false;

		/**
		 * Whether the rays are traced through a hierarchical depth buffer instead of being marched linearly.
		 * The cost of a ray then barely depends on its length, so long shadows stay cheap. The rays are exact,
		 * `quality` and the temporal filtering are not used. Requires a WebGPU backend.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.hiZ = false;

		/**
		 * Whether a ray traced with `hiZ` stops at any occluder instead of the closest one, which takes fewer steps.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.anyHit = false;

		/**
		 * The maximum number of steps of a ray when `hiZ` is used.
		 *
		 * @type {UniformNode<int>}
		 * @default 64
		 */
		this.maxSteps = uniform( 64, 'int' );

		/**
		 * The `updateBeforeType` is set to `NodeUpdateType.FRAME` since the node renders
		 * its effect once per frame in `updateBefore()`.
		 *
		 * @type {string}
		 * @default 'frame'
		 */
		this.updateBeforeType = NodeUpdateType.FRAME;

		// private uniforms

		/**
		 * Represents the view matrix of the scene's camera.
		 *
		 * @private
		 * @type {UniformNode<mat4>}
		 */
		this._cameraViewMatrix = uniform( camera.matrixWorldInverse );

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

		/**
		 * The resolution of the pass.
		 *
		 * @private
		 * @type {UniformNode<vec2>}
		 */
		this._resolution = uniform( new Vector2() );

		/**
		 * Temporal offset added to the initial ray step.
		 *
		 * @private
		 * @type {UniformNode<float>}
		 */
		this._temporalOffset = uniform( 0 );

		/**
		 * The frame ID use when temporal filtering is enabled.
		 *
		 * @private
		 * @type {UniformNode<uint>}
		 */
		this._frameId = uniform( 0 );

		/**
		 * A reference to the scene's main light.
		 *
		 * @private
		 * @type {DirectionalLight}
		 */
		this._mainLight = mainLight;

		/**
		 * The camera the scene is rendered with.
		 *
		 * @private
		 * @type {Camera}
		 */
		this._camera = camera;

		/**
		 * The render target the SSS is rendered into.
		 *
		 * @private
		 * @type {RenderTarget}
		 */
		this._sssRenderTarget = new RenderTarget( 1, 1, { depthBuffer: false, format: RedFormat, type: UnsignedByteType } );
		this._sssRenderTarget.texture.name = 'SSS';

		/**
		 * The material that is used to render the effect.
		 *
		 * @private
		 * @type {NodeMaterial}
		 */
		this._material = new NodeMaterial();
		this._material.name = 'SSS';

		/**
		 * The result of the effect is represented as a separate texture node.
		 *
		 * @private
		 * @type {PassTextureNode}
		 */
		this._textureNode = passTexture( this, this._sssRenderTarget.texture );

		/**
		 * The hierarchical depth buffer, created when `hiZ` is used.
		 *
		 * @private
		 * @type {?HiZNode}
		 */
		this._hiZNode = null;

		this._fragmentFn = null;
		this._builtKey = '';

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
	 * Sets the size of the effect.
	 *
	 * @param {number} width - The width of the effect.
	 * @param {number} height - The height of the effect.
	 */
	setSize( width, height ) {

		width = Math.round( this.resolutionScale * width );
		height = Math.round( this.resolutionScale * height );

		this._resolution.value.set( width, height );
		this._sssRenderTarget.setSize( width, height );

	}

	/**
	 * This method is used to render the effect once per frame.
	 *
	 * @param {NodeFrame} frame - The current node frame.
	 */
	updateBefore( frame ) {

		const { renderer } = frame;

		// the material reads these properties when it is built

		const key = `${ this.hiZ },${ this.anyHit }`;

		if ( this._fragmentFn !== null && key !== this._builtKey ) {

			this._builtKey = key;
			this._material.fragmentNode = this._fragmentFn();
			this._material.needsUpdate = true;

		}

		if ( this.hiZ === true ) {

			if ( this._hiZNode === null ) this._hiZNode = hiZ( this.depthNode, { nearest: true } );

			this._hiZNode.updateBefore( frame );

		}

		_rendererState = RendererUtils.resetRendererState( renderer, _rendererState );

		//

		const size = renderer.getDrawingBufferSize( _size );
		this.setSize( size.width, size.height );

		// update temporal uniforms

		if ( this.useTemporalFiltering === true ) {

			const frameId = frame.frameId;

			this._temporalOffset.value = _spatialOffsets[ frameId % 4 ];
			this._frameId = frame.frameId;

		} else {

			this._temporalOffset.value = 0;
			this._frameId = 0;

		}

		//

		_quadMesh.material = this._material;
		_quadMesh.name = 'SSS';

		// clear

		renderer.setClearColor( 0xffffff, 1 );

		// sss

		renderer.setRenderTarget( this._sssRenderTarget );
		_quadMesh.render( renderer );

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

		const sampleDepth = ( uv ) => projectionDepth( this.depthNode.sample( uv ).r, this._cameraNear, this._cameraFar );

		const sampleNormal = ( uv ) => this.normalNode !== null ? this.normalNode.sample( uv ).rgb.normalize() : getNormalFromDepth( uv, this.depthNode.value, this._cameraProjectionMatrixInverse );

		const sss = Fn( () => {

			const depth = sampleDepth( uvNode ).toVar();
			isBackgroundDepth( depth ).discard();

			// compute ray position and direction (in view-space)

			const rayStartPosition = getViewPosition( uvNode, depth, this._cameraProjectionMatrixInverse ).toVar( 'rayStartPosition' );
			const rayDirection = this._cameraViewMatrix.transformDirection( lightPosition( this._mainLight ).sub( lightTargetPosition( this._mainLight ) ) ).toConst( 'rayDirection' );
			const rayEndPosition = rayStartPosition.add( rayDirection.mul( this.maxDistance ) ).toVar( 'rayEndPosition' );

			// the ray ends at the near plane

			If( rayEndPosition.z.greaterThan( this._cameraNear.negate() ), () => {

				const t = this._cameraNear.negate().sub( rayStartPosition.z ).div( rayDirection.z ).mul( 0.999 );

				rayEndPosition.assign( rayStartPosition.add( rayDirection.mul( t ) ) );

			} );

			// the viewZ of the ray, its parameter is linear in screen space

			const rayZ = this._camera.isPerspectiveCamera === true
				? ( s ) => mix( rayStartPosition.z.reciprocal(), rayEndPosition.z.reciprocal(), s ).reciprocal()
				: ( s ) => mix( rayStartPosition.z, rayEndPosition.z, s );

			// an occluder is above the plane of the receiver by more than a pixel, otherwise a surface at a grazing
			// angle shadows itself: the depth of a pixel is constant, while its surface spans a range of depths

			const normal = sampleNormal( uvNode ).toVar();
			const facing = normal.dot( rayStartPosition ).lessThan( 0 ).select( normal, normal.negate() ).toConst();
			const pixelSize = float( 2 ).div( this._cameraProjectionMatrix.element( 1 ).element( 1 ).mul( this._resolution.y ) );

			const occludes = ( uv, raw ) => {

				const position = getViewPosition( uv, projectionDepth( raw, this._cameraNear, this._cameraFar ), this._cameraProjectionMatrixInverse ).toConst();
				const bias = this._camera.isPerspectiveCamera === true ? pixelSize.mul( position.z.negate() ) : pixelSize;

				return position.sub( rayStartPosition ).dot( facing ).greaterThan( bias );

			};

			if ( this.hiZ === true ) {

				// a surface facing away from the light shadows itself, no ray is traced

				const value = this.shadowIntensity.oneMinus().toVar();

				If( facing.dot( rayDirection ).greaterThan( 0 ), () => {

					value.assign( this._setupHiZ( builder, uvNode, depth, rayEndPosition, rayZ, occludes ) );

				} );

				return value;

			}

			// d0 and d1 are the start and maximum points of the ray in screen space
			const d0 = screenCoordinate.xy.toVar();
			const d1 = getScreenPosition( rayEndPosition, this._cameraProjectionMatrix ).mul( this._resolution ).toVar();

			// below variables are used to control the raymarching process

			// total length of the ray
			const totalLen = d1.sub( d0 ).length().toVar();

			// offset in x and y direction
			const xLen = d1.x.sub( d0.x ).toVar();
			const yLen = d1.y.sub( d0.y ).toVar();

			// determine the larger delta
			// The larger difference will help to determine how much to travel in the X and Y direction each iteration and
			// how many iterations are needed to travel the entire ray
			const totalStep = int( max( abs( xLen ), abs( yLen ) ).mul( this.quality.clamp() ) ).toConst();

			// step sizes in the x and y directions
			const xSpan = xLen.div( totalStep ).toVar();
			const ySpan = yLen.div( totalStep ).toVar();

			// compute noise based ray offset
			const noise = interleavedGradientNoise( screenCoordinate );
			const offset = fract( noise.add( this._temporalOffset ) ).add( rand( uvNode.add( this._frameId ) ) ).toConst( 'offset' );

			const result = float( 1 ).toVar();

			Loop( totalStep, ( { i } ) => {

				// advance on the ray by computing a new position in screen coordinates
				const xy = vec2( d0.x.add( xSpan.mul( float( i ).add( offset ) ) ), d0.y.add( ySpan.mul( float( i ).add( offset ) ) ) ).toVar();

				// stop processing if the new position lies outside of the screen
				If( xy.x.lessThan( 0 ).or( xy.x.greaterThan( this._resolution.x ) ).or( xy.y.lessThan( 0 ) ).or( xy.y.greaterThan( this._resolution.y ) ), () => {

					Break();

				} );

				// compute new uv and viewZ for the next fragment

				const uvNode = xy.div( this._resolution );
				const fragmentViewZ = depthToViewZ( this.depthNode.sample( uvNode ).r, this._cameraNear, this._cameraFar, this._camera ).toConst( 'fragmentViewZ' );

				const s = xy.sub( d0 ).length().div( totalLen ).toVar();
				const depthDelta = rayZ( s ).sub( fragmentViewZ ).negate(); // Port note: viewZ values are negative in three

				// check if the camera can't "see" the ray (ray depth must be larger than the camera depth, so positive depth_delta)

				If( depthDelta.greaterThan( 0 ).and( depthDelta.lessThan( this.thickness ) ), () => {

					// mark as occluded

					result.assign( this.shadowIntensity.oneMinus() );

					Break();

				} );


			} );

			return result;

		} );

		this._material.contextNode = context( builder.getSharedContext() );
		this._fragmentFn = sss;
		this._builtKey = `${ this.hiZ },${ this.anyHit }`;
		this._material.fragmentNode = sss();
		this._material.needsUpdate = true;

		return this._textureNode;

	}

	/**
	 * Traces the shadow ray through the hierarchical depth buffer.
	 *
	 * @private
	 * @param {NodeBuilder} builder - The current node builder.
	 * @param {Node<vec2>} uvNode - The uv of the fragment.
	 * @param {Node<float>} depth - The depth of the fragment.
	 * @param {Node<vec3>} rayEndPosition - The view position of the end of the ray.
	 * @param {Function} rayZ - Returns the viewZ of the ray at a parameter.
	 * @param {Function} occludes - Returns whether the surface at a uv and a raw depth can occlude the fragment.
	 * @return {Node<float>} The shadow factor.
	 */
	_setupHiZ( builder, uvNode, depth, rayEndPosition, rayZ, occludes ) {

		if ( this._hiZNode === null ) this._hiZNode = hiZ( this.depthNode, { nearest: true } );

		const near = this._cameraNear;
		const far = this._cameraFar;
		const perspective = this._camera.isPerspectiveCamera === true;
		const reversed = builder.renderer.reversedDepthBuffer === true && builder.renderer.logarithmicDepthBuffer !== true;

		const toDepth = perspective
			? ( reversed ? viewZToReversedPerspectiveDepth : viewZToPerspectiveDepth )
			: ( reversed ? viewZToReversedOrthographicDepth : viewZToOrthographicDepth );

		const origin = vec3( uvNode, depth ).toConst();
		const end = vec3( getScreenPosition( rayEndPosition, this._cameraProjectionMatrix ), toDepth( rayEndPosition.z, near, far ) ).toConst();
		const delta = end.xy.sub( origin.xy ).toConst();

		// the distance of the ray behind a surface, an occluder is at most `thickness` in front of the ray

		const behind = ( s, raw ) => depthToViewZ( raw, near, far, this._camera ).sub( rayZ( s ) );
		const parameter = ( uv ) => uv.sub( origin.xy ).dot( delta ).div( delta.dot( delta ).max( 1e-12 ) ).clamp();

		const result = traceHiZ( this._hiZNode, origin, end, {
			reversedDepth: reversed,
			start: delta.mul( this._hiZNode.size ).length().max( 1 ).reciprocal(), // skip the pixel of the fragment
			maxSteps: this.maxSteps,
			depthTolerance: 2e-7,
			projectDepth: ( raw ) => projectionDepth( raw, near, far ),
			isHidden: ( uvIn, uvOut, raw ) => behind( parameter( uvIn ), raw ).greaterThan( this.thickness ).and( behind( parameter( uvOut ), raw ).greaterThan( this.thickness ) ),
			acceptHit: ( uv, s, raw ) => behind( s, raw ).lessThan( this.thickness ).and( occludes( uv, raw ) ),
			anyHit: this.anyHit,
			passRejected: true
		} );

		const value = float( 1 ).toVar();

		If( result.hit, () => {

			value.assign( this.shadowIntensity.oneMinus() );

		} );

		return value;

	}

	/**
	 * Frees internal resources. This method should be called
	 * when the effect is no longer required.
	 */
	dispose() {

		super.dispose();

		this._sssRenderTarget.dispose();

		this._material.dispose();

		if ( this._hiZNode !== null ) this._hiZNode.dispose();

	}

}

export default SSSNode;

/**
 * TSL function for creating a SSS effect.
 *
 * @tsl
 * @function
 * @param {TextureNode} depthNode - A texture node that represents the scene's depth.
 * @param {Camera} camera - The camera the scene is rendered with.
 * @param {DirectionalLight} mainLight - The main directional light of the scene.
 * @returns {SSSNode}
 */
export const sss = ( depthNode, camera, mainLight ) => new SSSNode( depthNode, camera, mainLight );

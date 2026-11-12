import { Matrix4, Vector2 } from 'three/webgpu';
import { Fn, float, getScreenPosition, getViewPosition, isBackgroundDepth, ivec2, rtt, sample, textureSize, uniform, vec2, vec3, vec4 } from 'three/tsl';

/**
 * Samples previous-frame scene radiance at current-frame UV coordinates.
 *
 * The returned sample node can be shared by SSR, SSGI, or other screen-space effects.
 * RGB contains radiance and alpha contains valid bilinear support. Missing, off-screen,
 * background, resized, or depth-inconsistent history returns zero radiance.
 *
 * `camera` reprojects each query with camera matrices and assumes static geometry.
 * `velocity` reprojects each query with the scene pass's velocity. `buffer` performs
 * velocity reprojection once per frame into a texture shared by all consumers.
 *
 * History is requested lazily from the scene pass. Camera state is captured once per
 * frame while the sampler renders, including any projection jitter. Velocity jitter
 * correction assumes a perspective projection. Merely constructing
 * this sampler does not render a pass or request previous scene textures.
 *
 * @param {PassNode} passNode - The scene pass, with a `velocity` MRT output for velocity/buffer modes.
 * @param {Object} [options={}] - Options.
 * @param {('camera'|'velocity'|'buffer')} [options.mode='velocity'] - The reprojection method.
 * @param {?SampleNode} [options.previousViewDepthNode=null] - A scalar sampler of the surface's previous view-space Z, including object motion. Without it, depth validation assumes static geometry, even in velocity mode.
 * @param {number} [options.resolutionScale=1] - Buffer resolution scale, used only in buffer mode.
 * @return {SampleNode} Radiance sampler. Dispose it to release its optional reprojection buffer; the scene pass owns the history textures.
 * @three_import import { reprojectedRadiance } from 'three/addons/tsl/display/ReprojectedRadiance.js';
 */
export function reprojectedRadiance( passNode, { mode = 'velocity', previousViewDepthNode = null, resolutionScale = 1 } = {} ) {

	if ( ! [ 'camera', 'velocity', 'buffer' ].includes( mode ) ) {

		throw new Error( 'reprojectedRadiance: Unknown mode.' );

	}

	if ( ! Number.isFinite( resolutionScale ) || resolutionScale <= 0 ) {

		throw new Error( 'reprojectedRadiance: resolutionScale must be positive.' );

	}

	const camera = passNode.camera;
	const cameraWorld = uniform( camera.matrixWorld );
	const projectionInverse = uniform( camera.projectionMatrixInverse );
	const previousView = uniform( new Matrix4() );
	const previousProjection = uniform( new Matrix4() );
	const previousProjectionInverse = uniform( new Matrix4() );
	const jitterDelta = uniform( new Vector2() );

	const capturedView = new Matrix4();
	const capturedProjection = new Matrix4();
	const capturedSize = new Vector2();
	const capturedJitter = new Vector2();
	const jitter = new Vector2();
	let capturedFrame = - 1;

	const historyValid = uniform( false ).onFrameUpdate( ( { frameId } ) => {

		const { width, height } = passNode.renderTarget;
		const projection = camera.projectionMatrix.elements;
		jitter.set( - projection[ 8 ] * 0.5, projection[ 9 ] * 0.5 );

		const valid = frameId === capturedFrame + 1 && capturedSize.x === width && capturedSize.y === height;
		previousView.value.copy( capturedView );
		previousProjection.value.copy( capturedProjection );
		previousProjectionInverse.value.copy( capturedProjection ).invert();
		jitterDelta.value.copy( capturedJitter ).sub( jitter );

		capturedView.copy( camera.matrixWorldInverse );
		capturedProjection.copy( camera.projectionMatrix );
		capturedSize.set( width, height );
		capturedJitter.copy( jitter );
		capturedFrame = frameId;

		return valid;

	} );

	const reproject = sample( uv => Fn( () => {

		const previousColor = passNode.getPreviousTextureNode( 'output' );
		const previousDepth = passNode.getPreviousTextureNode( 'depth' );
		const depth = passNode.getTextureNode( 'depth' ).sample( uv ).r.toConst();
		let historyUV, expectedZ;

		if ( mode === 'camera' || previousViewDepthNode === null ) {

			const viewPosition = getViewPosition( uv, depth, projectionInverse );
			const previousPosition = previousView.mul( cameraWorld.mul( vec4( viewPosition, 1 ) ) ).xyz.toConst();
			historyUV = getScreenPosition( previousPosition, previousProjection ).toConst();
			expectedZ = previousPosition.z;

		}

		if ( mode !== 'camera' ) {

			// A point load keeps motion on a single surface at silhouettes.
			const velocity = passNode.getTextureNode( 'velocity' );
			const size = ivec2( textureSize( velocity ) );
			const texel = ivec2( uv.mul( vec2( size ) ) ).clamp( ivec2( 0 ), size.sub( 1 ) );
			const motion = velocity.load( texel ).xy;
			historyUV = uv.sub( motion.mul( vec2( 0.5, - 0.5 ) ) ).add( jitterDelta ).toConst();

			if ( previousViewDepthNode !== null ) expectedZ = previousViewDepthNode.sample( uv ).toConst();

		}

		const size = vec2( textureSize( previousDepth ) ).toConst();
		const pixel = historyUV.mul( size ).sub( 0.5 ).toConst();
		const base = pixel.floor();
		const fraction = pixel.fract();
		const inBounds = p => p.greaterThanEqual( 0 ).all().and( p.lessThanEqual( 1 ).all() );
		const valid = historyValid.and( inBounds( historyUV ) ).and( isBackgroundDepth( depth ).not() )
			.and( expectedZ.lessThan( camera.near * - 1 ) );

		const pixelDepth = previousProjectionInverse.element( 1 ).y.abs().mul( expectedZ.abs() ).mul( 3 ).div( size.y );
		const tolerance = pixelDepth.max( expectedZ.abs().mul( 0.002 ) ).max( 0.001 );
		const radiance = vec3( 0 ).toVar();
		const support = float( 0 ).toVar();

		// Validate every bilinear tap before interpolation to avoid unrelated surfaces.
		for ( let y = 0; y <= 1; y ++ ) {

			for ( let x = 0; x <= 1; x ++ ) {

				const tapUV = base.add( vec2( x, y ) ).add( 0.5 ).div( size ).toConst();
				const tapDepth = previousDepth.sample( tapUV ).r;
				const viewZ = getViewPosition( tapUV, tapDepth, previousProjectionInverse ).z;
				const accepted = valid.and( inBounds( tapUV ) ).and( isBackgroundDepth( tapDepth ).not() )
					.and( viewZ.sub( expectedZ ).abs().lessThanEqual( tolerance ) );
				const weightX = x === 0 ? fraction.x.oneMinus() : fraction.x;
				const weightY = y === 0 ? fraction.y.oneMinus() : fraction.y;
				const weight = weightX.mul( weightY ).mul( float( accepted ) ).toConst();
				radiance.addAssign( previousColor.sample( tapUV ).rgb.mul( weight ) );
				support.addAssign( weight );

			}

		}

		return vec4( radiance.div( support.max( 1e-6 ) ), support );

	} )() );

	if ( mode !== 'buffer' ) return reproject;

	const buffer = rtt( sample( uv => Fn( () => {

		const history = reproject.sample( uv ).toConst();
		return vec4( history.rgb.mul( history.a ), history.a );

	} )() ), null, null, { resolutionScale, depthBuffer: false } );

	const result = sample( uv => Fn( () => {

		const history = buffer.sample( uv ).toConst();
		return history.a.greaterThan( 0.01 ).select( vec4( history.rgb.div( history.a ), history.a ), vec4( 0 ) );

	} )() );
	result.addEventListener( 'dispose', () => buffer.dispose() );

	return result;

}

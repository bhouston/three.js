import { HalfFloatType, Matrix4, RenderTarget, Vector2 } from 'three/webgpu';
import { Fn, float, getViewPosition, isBackgroundDepth, ivec2, sample, texture, textureSize, uniform, vec2, vec3, vec4 } from 'three/tsl';

/**
 * Reprojects previous-frame beauty for SSR, validating the depth of each history tap.
 * The velocity input stores motion in XY and the surface's previous view-space Z in Z.
 * Call prepare() before rendering and capture() after the beauty pass has completed.
 * The separate color history keeps the beauty pass out of its own dependency graph.
 *
 * @param {Camera} camera - The scene camera.
 * @param {PassNode} scenePass - The current geometry pass, including depth.
 * @param {Node<vec4>} scenePassVelocity - Motion and previous view-space depth.
 * @returns {Object} The source node and its history lifecycle methods.
 */
export function ssrHistory( camera, scenePass, scenePassVelocity ) {

	const scenePassDepth = scenePass.getTextureNode( 'depth' );

	const previousJitter = new Vector2();
	const currentJitter = new Vector2();
	const currentProjectionInverse = new Matrix4();
	const previousProjectionInverse = uniform( new Matrix4() );
	const historySize = uniform( new Vector2() );
	const jitterDelta = uniform( new Vector2() );
	const historyValid = uniform( false ).onRenderUpdate( () => {

		// Capture the camera while TRAA jitter is applied, in whichever pass reprojects history.
		const e = camera.projectionMatrix.elements;
		currentJitter.set( - e[ 8 ] * 0.5, e[ 9 ] * 0.5 );
		currentProjectionInverse.copy( camera.projectionMatrixInverse );
		jitterDelta.value.copy( previousJitter ).sub( currentJitter );
		return historyValid.value;

	} );
	const beautyHistory = new RenderTarget( 1, 1, { type: HalfFloatType, depthBuffer: false } );
	const previousColor = texture( beautyHistory.texture );
	const previousDepth = scenePass.getPreviousTextureNode( 'depth' );

	const source = sample( uv => Fn( () => {

		// Reproject both camera and object motion. Read motion and previous view
		// depth from the same texel to keep silhouettes from mixing surfaces.
		const motionSize = ivec2( textureSize( scenePassVelocity ) );
		const motionTexel = ivec2( uv.mul( vec2( motionSize ) ) ).clamp( ivec2( 0 ), motionSize.sub( 1 ) );
		const motion = scenePassVelocity.load( motionTexel ).toConst();
		const historyUV = uv.sub( motion.xy.mul( vec2( 0.5, - 0.5 ) ) ).add( jitterDelta ).toConst();
		const expectedZ = motion.z;

		const size = vec2( textureSize( previousDepth ) );
		const pixel = historyUV.mul( size ).sub( 0.5 );
		const base = pixel.floor();
		const fraction = pixel.fract();
		const inBounds = p => p.greaterThanEqual( 0 ).all().and( p.lessThanEqual( 1 ).all() );
		const valid = historyValid.and( inBounds( historyUV ) )
			.and( size.equal( historySize ).all() )
			.and( vec2( textureSize( scenePassDepth ) ).equal( historySize ).all() )
			.and( isBackgroundDepth( scenePassDepth.sample( uv ).r ).not() )
			.and( expectedZ.lessThan( camera.near * - 1 ) );

		// Allow a small pixel footprint for depth variation on sloping surfaces, plus half-float rounding.
		const pixelDepth = previousProjectionInverse.element( 1 ).y.abs().mul( expectedZ.abs() ).mul( 3 ).div( size.y );
		const tolerance = pixelDepth.max( expectedZ.abs().mul( 0.002 ) ).max( 0.001 );
		const radiance = vec3( 0 ).toVar();
		const support = float( 0 ).toVar();

		// Validate each bilinear tap so interpolation cannot pull color across an occlusion edge.
		for ( let y = 0; y <= 1; y ++ ) {

			for ( let x = 0; x <= 1; x ++ ) {

				const tapUV = base.add( vec2( x, y ) ).add( 0.5 ).div( size ).toConst();
				const depth = previousDepth.sample( tapUV ).r;
				const viewZ = getViewPosition( tapUV, depth, previousProjectionInverse ).z;
				const accepted = valid.and( inBounds( tapUV ) ).and( isBackgroundDepth( depth ).not() )
					.and( viewZ.sub( expectedZ ).abs().lessThanEqual( tolerance ) );
				const weightX = x === 0 ? fraction.x.oneMinus() : fraction.x;
				const weightY = y === 0 ? fraction.y.oneMinus() : fraction.y;
				const weight = weightX.mul( weightY ).mul( accepted ).toConst();
				radiance.addAssign( previousColor.sample( tapUV ).rgb.mul( weight ) );
				support.addAssign( weight );

			}

		}

		// Missing history contributes no radiance; alpha records valid bilinear support.
		return vec4( radiance.div( support.max( 1e-6 ) ), support );

	} )() );


	return {
		source,

		prepare( renderer ) {

			// Resize history before rendering so an old-sized texture is never read as valid.
			const { width, height } = renderer.domElement;

			if ( beautyHistory.width !== width || beautyHistory.height !== height ) {

				beautyHistory.setSize( width, height );
				historyValid.value = false;

			}

			renderer.initRenderTarget( beautyHistory );

		},

		capture( renderer, beautyTexture ) {

			const { width, height } = renderer.domElement;
			renderer.copyTextureToTexture( beautyTexture, beautyHistory.texture );

			previousJitter.copy( currentJitter );
			previousProjectionInverse.value.copy( currentProjectionInverse );
			historySize.value.set( width, height );
			historyValid.value = true;

		},

		invalidate() {

			historyValid.value = false;

		},

		dispose() {

			beautyHistory.dispose();

		}
	};

}

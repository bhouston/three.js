import { Fn, float, getViewPosition, sample, textureSize, uniform, vec2, vec4 } from 'three/tsl';

/**
 * Experimental joint bilateral reconstruction kernels. The default options
 * reproduce jointBilateralUpsample; each option changes one part of the filter.
 *
 * Reference: https://johanneskopf.de/publications/jbu/
 *
 * @param {Node<vec4>} inputNode - Reduced resolution color.
 * @param {TextureNode} lowDepthNode - Reduced resolution depth.
 * @param {?Node<vec3>} lowNormalNode - Reduced resolution normalized view normals; unused with depth-only guidance.
 * @param {Node<float>} depthNode - Full resolution depth.
 * @param {Node<vec3>} normalNode - Full resolution normalized view normals.
 * @param {Camera} camera - Scene camera.
 * @param {Object} [options={}] - Compile-time filter configuration.
 * @param {string} [options.guidance='normal-depth'] - 'depth-only' skips low resolution normal reads.
 * @param {string} [options.depthKernel='exponential'] - 'rational' approximates the Gaussian without exp().
 * @param {string} [options.reconstruction='bilinear'] - 'gaussian' uses nine taps; 'sharp' uses a sharpened four-tap tent.
 * @param {number} [options.depthTolerance=0.005] - Plane distance threshold relative to view depth.
 * @param {number} [options.normalPower=8] - Normal rejection exponent, a power of two.
 * @param {Node<vec2>} [options.size] - Optional reduced resolution grid dimensions.
 * @param {boolean} [options.coverage=false] - Filter coverage-weighted RGB separately from alpha.
 * @returns {SampleNode} Reconstructed color.
 */
export const researchJointBilateralUpsample = ( inputNode, lowDepthNode, lowNormalNode, depthNode, normalNode, camera, options = {} ) => {

	const {
		guidance = 'normal-depth', depthKernel = 'exponential', reconstruction = 'bilinear',
		depthTolerance = 0.005, normalPower = 8, coverage = false, size: gridSize = null
	} = options;

	const projectionInverse = uniform( camera.projectionMatrixInverse );
	const wide = reconstruction === 'gaussian';

	return sample( coordinate => Fn( () => {

		const size = vec2( gridSize !== null ? gridSize : textureSize( lowDepthNode ) );
		const pixel = coordinate.mul( size ).sub( 0.5 );
		const base = ( wide ? pixel.round() : pixel.floor() ).toConst();
		const fraction = pixel.fract();
		const position = getViewPosition( coordinate, depthNode.sample( coordinate ).r, projectionInverse ).toConst();
		const n = normalNode.sample( coordinate ).toConst();
		const tolerance = position.z.abs().mul( depthTolerance ).max( 0.001 );

		const gathered = ! wide && gridSize === null ? lowDepthNode.gather().sample( base.add( 1 ).div( size ) ).toConst() : null;
		const tapDepths = gathered !== null ? [ gathered.w, gathered.z, gathered.x, gathered.y ] : null;
		const sum = vec4( 0 ).toVar();
		const total = float( 0 ).toVar();

		for ( let y = wide ? - 1 : 0; y <= 1; y ++ ) {

			for ( let x = wide ? - 1 : 0; x <= 1; x ++ ) {

				const tapPixel = base.add( vec2( x, y ) );
				const tapUV = tapPixel.add( 0.5 ).div( size ).toConst();
				const tapDepth = tapDepths !== null ? tapDepths[ y * 2 + x ] : lowDepthNode.sample( tapUV ).r;
				const planeDistance = getViewPosition( tapUV, tapDepth, projectionInverse ).sub( position ).dot( n );
				const d2 = planeDistance.div( tolerance ).pow2().toConst();

				// The reciprocal of the degree-three Taylor polynomial of exp(d²)
				// matches exp(-d²) around zero without a transcendental operation.
				const depthWeight = depthKernel === 'rational'
					? float( 1 ).div( d2.mul( 1 / 6 ).add( 0.5 ).mul( d2 ).add( 1 ).mul( d2 ).add( 1 ) )
					: d2.negate().exp();

				let normalWeight = float( 1 );
				if ( guidance !== 'depth-only' ) {

					normalWeight = lowNormalNode.sample( tapUV ).dot( n ).max( 0 );
					for ( let power = 1; power < normalPower; power *= 2 ) normalWeight = normalWeight.toConst().pow2();

				}

				let spatialWeight;
				if ( wide ) {

					const delta = pixel.sub( tapPixel );
					spatialWeight = delta.dot( delta ).mul( - 2.29 ).exp();

				} else {

					const wx = x === 0 ? fraction.x.oneMinus() : fraction.x;
					const wy = y === 0 ? fraction.y.oneMinus() : fraction.y;
					// A squared tent narrows the reconstruction footprint while retaining
					// smooth interpolation and the same four color/guide reads.
					spatialWeight = wx.mul( wy );
					if ( reconstruction === 'sharp' ) spatialWeight = spatialWeight.pow2();

				}

				const weight = spatialWeight.mul( depthWeight.mul( normalWeight ).add( 1e-4 ) ).toConst();
				const value = inputNode.sample( tapUV );
				sum.addAssign( ( coverage ? vec4( value.rgb.mul( value.a ), value.a ) : value ).mul( weight ) );
				total.addAssign( weight );

			}

		}

		return coverage ? vec4( sum.rgb.div( sum.a.max( 1e-6 ) ), sum.a.div( total ) ) : sum.div( total );

	} )() );

};

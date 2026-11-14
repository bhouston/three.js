import { WebGPUCoordinateSystem } from 'three/webgpu';
import { Fn, float, sample, textureSize, uniform, vec2, vec4 } from 'three/tsl';

/**
 * Upsamples a reduced resolution effect to full resolution with a joint bilateral
 * filter: the four nearest low resolution texels are bilinearly weighted and then
 * rejected by view depth and normal differences against the full resolution guides,
 * so the result does not bleed across silhouettes.
 *
 * A small floor on the weights fades the result to plain bilinear interpolation when
 * no texel matches.
 *
 * All four channels are filtered alike by default. With the `coverage` option the
 * input is expected to store a signal in RGB and its coverage in alpha: RGB is then
 * averaged weighted by coverage, alpha by the filter weights alone.
 *
 * Reference: {@link https://johanneskopf.de/publications/jbu/}
 *
 * @tsl
 * @function
 * @param {Node<vec4>} inputNode - The low resolution texture to upsample.
 * @param {TextureNode} lowDepthNode - Depth at the input's resolution, in the red channel. Its size defines the texel grid.
 * @param {Node<vec3>} lowNormalNode - Normalized view-space normals at the input's resolution.
 * @param {Node<float>} depthNode - Full resolution depth, in the red channel.
 * @param {Node<vec3>} normalNode - Normalized full resolution view-space normals.
 * @param {Camera} camera - The camera the scene is rendered with.
 * @param {Object} [options={}] - The filter options.
 * @param {Node<float>|number} [options.depthTolerance=0.02] - The view depth difference, relative to the view depth, at which a texel is rejected.
 * @param {number} [options.normalPower=8] - How strongly a normal difference rejects a texel. Must be a power of two.
 * @param {boolean} [options.coverage=false] - Whether the input's alpha is the coverage of its RGB signal.
 * @returns {SampleNode} The upsampled result, samplable at any coordinate.
 */
export const jointBilateralUpsample = ( inputNode, lowDepthNode, lowNormalNode, depthNode, normalNode, camera, options = {} ) => {

	const { depthTolerance = 0.02, normalPower = 8, coverage = false } = options;

	const projectionInverse = uniform( camera.projectionMatrixInverse );

	return sample( coordinate => Fn( ( builder ) => {

		const size = vec2( textureSize( lowDepthNode ) );
		const pixel = coordinate.mul( size ).sub( 0.5 );
		const base = pixel.floor();
		const fraction = pixel.fract();

		// view z depends on depth alone, so only the z and w rows of the inverse projection are needed
		const zRow = vec2( projectionInverse.element( 2 ).z, projectionInverse.element( 3 ).z ).toConst();
		const wRow = vec2( projectionInverse.element( 2 ).w, projectionInverse.element( 3 ).w ).toConst();
		const viewZ = depth => {

			if ( builder.renderer.coordinateSystem !== WebGPUCoordinateSystem && builder.renderer.reversedDepthBuffer !== true ) depth = depth.mul( 2 ).sub( 1 );
			return depth.mul( zRow.x ).add( zRow.y ).div( depth.mul( wRow.x ).add( wRow.y ) );

		};

		const z = viewZ( depthNode.sample( coordinate ).r ).toConst();
		const n = normalNode.sample( coordinate ).toConst();
		const tolerance = z.abs().mul( depthTolerance ).max( 0.001 );

		// the four low resolution view depths from one fetch, ordered like textureGather()
		const gathered = viewZ( lowDepthNode.gather().sample( base.add( 1 ).div( size ) ) ).toConst();
		const tapZ = [ gathered.w, gathered.z, gathered.x, gathered.y ];

		const sum = vec4( 0 ).toVar();
		const total = float( 0 ).toVar();

		for ( let y = 0; y <= 1; y ++ ) {

			for ( let x = 0; x <= 1; x ++ ) {

				const tapUV = base.add( vec2( x, y ) ).add( 0.5 ).div( size ).toConst();
				const depthWeight = tapZ[ y * 2 + x ].sub( z ).div( tolerance ).pow2().negate().exp();
				let normalWeight = lowNormalNode.sample( tapUV ).dot( n ).max( 0 );
				for ( let power = 1; power < normalPower; power *= 2 ) normalWeight = normalWeight.toConst().pow2();
				const wx = x === 0 ? fraction.x.oneMinus() : fraction.x;
				const wy = y === 0 ? fraction.y.oneMinus() : fraction.y;
				const weight = wx.mul( wy ).mul( depthWeight.mul( normalWeight ).add( 1e-4 ) ).toConst();
				const value = inputNode.sample( tapUV );
				sum.addAssign( ( coverage ? vec4( value.rgb.mul( value.a ), value.a ) : value ).mul( weight ) );
				total.addAssign( weight );

			}

		}

		return coverage ? vec4( sum.rgb.div( sum.a.max( 1e-6 ) ), sum.a.div( total ) ) : sum.div( total );

	} )() );

};

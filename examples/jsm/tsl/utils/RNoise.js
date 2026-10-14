import { float, Fn, fract, int, vec2, vec4 } from 'three/tsl';

/**
 * Returns a TSL function that samples texture-free analytic R⁴ noise.
 * The sample index tile-shifts a 32×32 pattern with an R² sequence.
 * Each pixel contains a four-dimensional R-sequence sample.
 *
 * @param {UniformNode<Vector2>} resolution
 * @param {number} [seed=0] - Offsets the tile shift and the R⁴ sequence phase.
 */
export function bindAnalyticNoise( resolution, seed = 0 ) {

	const seedOffset = int( seed );

	const r4 = ( coords ) => {

		// Generalized golden ratio for four dimensions: P^5 = P + 1.
		const P = 1.1673039782614187;
		const t = coords.x.add( coords.y.mul( 32 ) ).add( float( seed ) );

		return fract( vec4( 1 / P, 1 / P ** 2, 1 / P ** 3, 1 / P ** 4 ).mul( t ).add( 0.5 ) );

	};

	return Fn( ( [ uvCoord, sampleIndex ] ) => {

		const index = int( sampleIndex ).add( seedOffset );
		const noise = vec4().toVar();

		const tileSize = float( 32 );

		const screenPixel = uvCoord.mul( resolution ).floor();
		const offset = fract( vec2(
			float( index ).mul( 0.7548776662 ),
			float( index ).mul( 0.5698402910 )
		) ).mul( tileSize ).floor();
		const coords = screenPixel.add( offset ).mod( tileSize );

		noise.assign( r4( coords ) );

		return noise;

	} );

}

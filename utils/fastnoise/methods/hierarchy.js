/**
 * Randomized hierarchical ordered dither with digital temporal stratification.
 *
 * Bayer's recursively dispersed ordering supplies inexpensive spatial rejection
 * of low frequencies. Independent quadrant reflections scramble its hierarchy;
 * this is an ordered-dither construction, not an optimized blue-noise spectrum.
 * See Bayer, "An optimum method for two-level rendition of continuous-tone
 * pictures" (1973), and Owen, "Randomly Permuted (t,m,s)-Nets and (t,s)-Sequences"
 * (1995) for the hierarchical scrambling / stratification ideas.
 *
 * Reversed temporal bits XOR the most significant rank bits. Every pixel visits
 * every temporal stratum exactly once per cycle, and aligned power-of-two time
 * blocks stratify the corresponding number of coarse bins. Every frame also
 * retains the same complete spatial rank histogram. There is no iterative solve.
 *
 * Tradeoff: dyadic spatial motifs and periodic temporal correlations remain,
 * especially at binary thresholds. This is deliberately not claimed to be STBN:
 * Wolfe et al., https://arxiv.org/abs/2112.09629, motivate evaluating the joint
 * spatial/temporal filtered error rather than inferring it from stratification.
 */

function hash( value ) {

	value = Math.imul( value ^ ( value >>> 16 ), 0x7feb352d );
	value = Math.imul( value ^ ( value >>> 15 ), 0x846ca68b );
	return ( value ^ ( value >>> 16 ) ) >>> 0;

}

function isPowerOfTwo( value ) {

	return Number.isInteger( value ) && value > 0 && ( value & ( value - 1 ) ) === 0;

}

export function generate( { size = 128, depth = 32, seed = 1 } = {} ) {

	// Keep integer ranks exact and Float32 midpoints strictly inside (0, 1).
	if ( ! isPowerOfTwo( size ) || size > 1024 ) {

		throw new RangeError( 'size must be a power of two between 1 and 1024.' );

	}

	if ( ! isPowerOfTwo( depth ) || depth > size * size || depth > 1024 ) {

		throw new RangeError( 'depth must be a power of two no larger than size squared or 1024.' );

	}

	if ( ! Number.isInteger( seed ) || seed < 0 || seed > 0xffffffff ) {

		throw new RangeError( 'seed must be an unsigned 32-bit integer.' );

	}

	const count = size * size;
	const levels = Math.log2( size );
	const timeBits = Math.log2( depth );
	const ranks = new Uint32Array( count );
	const result = new Float32Array( count * depth );
	const scale = 1 / count;

	for ( let y = 0; y < size; y ++ ) {

		for ( let x = 0; x < size; x ++ ) {

			let rank = 0;

			for ( let level = 0; level < levels; level ++ ) {

				// Shared scrambling within each parent keeps every quadrant's
				// digit distinct. Fine spatial bits become coarse rank bits.
				const parent = ( x >>> ( level + 1 ) ) + size * ( y >>> ( level + 1 ) );
				const scramble = hash( seed ^ hash( parent + level * count ) ) & 3;
				const xb = ( x >>> level ) & 1;
				const yb = ( y >>> level ) & 1;
				const digit = ( ( xb ^ yb ) * 2 + yb ) ^ scramble;
				rank = ( rank << 2 ) | digit;

			}

			ranks[ y * size + x ] = rank;

		}

	}

	for ( let z = 0; z < depth; z ++ ) {

		let reversed = 0;
		for ( let bit = 0; bit < timeBits; bit ++ ) reversed = ( reversed << 1 ) | ( ( z >>> bit ) & 1 );
		const mask = reversed << ( levels * 2 - timeBits );
		const offset = z * count;

		for ( let i = 0; i < count; i ++ ) {

			result[ offset + i ] = ( ( ranks[ i ] ^ mask ) + 0.5 ) * scale;

		}

	}

	return result;

}

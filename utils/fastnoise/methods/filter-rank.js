/**
 * Alternating periodic high-pass filtering and uniform rank projection.
 * Inspired by the spectral shaping / sorting construction discussed in
 * https://bartwronski.com/2020/04/26/optimizing-blue-noise-dithering-backpropagation-through-fourier-transform-and-sorting/
 *
 * This is an experimental continuous-valued noise volume, not an optimized
 * threshold mask: a blue continuous spectrum does not guarantee blue spectra
 * for every threshold. The separable stencil can also leave directional bias.
 * Filtering is O(iterations * size^2 * depth); histogram ranking is approximate
 * (65536 score bins) but produces an exact uniform discrete marginal. Seeded
 * shuffled tie order avoids scan-order patterns within histogram bins.
 */
export function generate( { size = 128, depth = 32, seed = 1, iterations = 12, strength = 0.8 } = {} ) {

	if ( ! Number.isInteger( size ) || size < 1 || ! Number.isInteger( depth ) || depth < 1 || ! Number.isInteger( iterations ) || iterations < 0 || ! Number.isFinite( strength ) || strength < 0 || strength > 1 ) {

		throw new RangeError( 'Expected positive integer dimensions, nonnegative integer iterations, and strength in [0, 1].' );

	}

	const plane = size * size;
	const count = plane * depth;
	const values = new Float32Array( count );
	const scratch = new Float32Array( count );
	const spatial = new Float32Array( count );
	const order = new Uint32Array( count );
	const bins = new Uint16Array( count );
	const offsets = new Uint32Array( 65536 );
	let state = seed >>> 0;
	const random = () => {

		state = ( state + 0x6D2B79F5 ) >>> 0;
		let value = state;
		value = Math.imul( value ^ value >>> 15, value | 1 );
		value ^= value + Math.imul( value ^ value >>> 7, value | 61 );
		return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

	};

	for ( let i = 0; i < count; i ++ ) order[ i ] = i;
	for ( let i = count - 1; i > 0; i -- ) {

		const j = Math.floor( random() * ( i + 1 ) );
		const swap = order[ i ];
		order[ i ] = order[ j ];
		order[ j ] = swap;

	}

	for ( let i = 0; i < count; i ++ ) values[ order[ i ] ] = ( i + 0.5 ) / count;

	for ( let iteration = 0; iteration < iterations; iteration ++ ) {

		// Binomial low-pass in X and Y, with toroidal boundary conditions.
		for ( let z = 0; z < depth; z ++ ) {

			for ( let y = 0; y < size; y ++ ) {

				const row = z * plane + y * size;
				for ( let x = 0; x < size; x ++ ) {

					const i = row + x;
					scratch[ i ] = ( values[ row + ( x + size - 1 ) % size ] + 2 * values[ i ] + values[ row + ( x + 1 ) % size ] ) * 0.25;

				}

			}

			for ( let y = 0; y < size; y ++ ) {

				const row = z * plane + y * size;
				const before = z * plane + ( y + size - 1 ) % size * size;
				const after = z * plane + ( y + 1 ) % size * size;
				for ( let x = 0; x < size; x ++ ) spatial[ row + x ] = ( scratch[ before + x ] + 2 * scratch[ row + x ] + scratch[ after + x ] ) * 0.25;

			}

		}

		let minimum = Infinity;
		let maximum = - Infinity;
		for ( let z = 0; z < depth; z ++ ) {

			const before = ( z + depth - 1 ) % depth * plane;
			const after = ( z + 1 ) % depth * plane;
			for ( let p = 0; p < plane; p ++ ) {

				const i = z * plane + p;
				// Product of spatial and temporal high-pass operators. A singleton
				// time axis uses only spatial shaping instead of annihilating it.
				const high = values[ i ] - spatial[ i ];
				const temporal = depth === 1 ? high : ( 2 * high - values[ before + p ] + spatial[ before + p ] - values[ after + p ] + spatial[ after + p ] ) * 0.25;
				const score = ( 1 - strength ) * ( values[ i ] - 0.5 ) + strength * temporal;
				scratch[ i ] = score;
				minimum = Math.min( minimum, score );
				maximum = Math.max( maximum, score );

			}

		}

		offsets.fill( 0 );
		const scale = maximum > minimum ? 65535 / ( maximum - minimum ) : 0;
		for ( let i = 0; i < count; i ++ ) {

			const bin = Math.min( 65535, Math.max( 0, Math.floor( ( scratch[ i ] - minimum ) * scale ) ) );
			bins[ i ] = bin;
			offsets[ bin ] ++;

		}

		let total = 0;
		for ( let bin = 0; bin < offsets.length; bin ++ ) {

			const frequency = offsets[ bin ];
			offsets[ bin ] = total;
			total += frequency;

		}

		for ( let rank = 0; rank < count; rank ++ ) {

			const i = order[ rank ];
			values[ i ] = ( offsets[ bins[ i ] ] ++ + 0.5 ) / count;

		}

	}

	return values;

}

// Derived from PR116 lattice.js (54a6dfe336cd33fc1fe129e4e1e6b10900395832).
/**
 * Experimental spatially optimized Cranley-Patterson shifts plus a rank-1
 * temporal sequence. No stored texture, dependency, or browser API required.
 */
export function generate( { size = 32, seed = 1, passes = 8 } = {} ) {

	if ( ! [ 16, 32, 64, 128 ].includes( size ) || ! Number.isInteger( passes ) || passes < 0 ) {

		throw new RangeError( 'size must be 16, 32, 64, or 128 and passes must be a nonnegative integer' );

	}

	let state = seed >>> 0;
	function random() {

		state = ( state + 0x6d2b79f5 ) >>> 0;
		let value = state;
		value = Math.imul( value ^ value >>> 15, value | 1 );
		value ^= value + Math.imul( value ^ value >>> 7, value | 61 );
		return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

	}

	const count = size * size;
	const shifts = new Float32Array( count );
	for ( let i = 0; i < count; i ++ ) shifts[ i ] = ( i + 0.5 ) / count;
	for ( let i = count - 1; i > 0; i -- ) {

		const j = Math.floor( random() * ( i + 1 ) );
		const value = shifts[ i ];
		shifts[ i ] = shifts[ j ];
		shifts[ j ] = value;

	}

	if ( passes === 0 ) return shifts;

	// Repel similar circular phases in a Gaussian spatial neighborhood. A
	// circular objective remains unchanged by the common temporal rotation.
	const resolution = 2048;
	const potential = new Float64Array( resolution + 1 );
	for ( let i = 0; i <= resolution; i ++ ) {

		const distance = Math.min( i / resolution, 1 - i / resolution );
		potential[ i ] = Math.exp( - distance * distance / ( 2 * 0.12 * 0.12 ) );

	}

	const offsets = [];
	const weights = [];
	for ( let dy = - 2; dy <= 2; dy ++ ) {

		for ( let dx = - 2; dx <= 2; dx ++ ) {

			if ( dx === 0 && dy === 0 ) continue;
			offsets.push( [ dx, dy ] );
			weights.push( Math.exp( - ( dx * dx + dy * dy ) / 2 ) );

		}

	}

	// Precomputed neighbors avoid modulo operations in the optimization loop.
	const neighbors = new Uint32Array( count * offsets.length );
	for ( let p = 0; p < count; p ++ ) {

		const x = p % size;
		const y = Math.floor( p / size );
		for ( let k = 0; k < offsets.length; k ++ ) {

			const [ dx, dy ] = offsets[ k ];
			neighbors[ p * offsets.length + k ] = ( ( y + dy + size * 2 ) % size ) * size + ( x + dx + size * 2 ) % size;

		}

	}

	for ( let attempt = 0; attempt < count * passes; attempt ++ ) {

		const a = Math.floor( random() * count );
		const b = Math.floor( random() * count );
		const av = shifts[ a ], bv = shifts[ b ];
		let delta = 0;
		for ( let k = 0; k < offsets.length; k ++ ) {

			const an = neighbors[ a * offsets.length + k ];
			const bn = neighbors[ b * offsets.length + k ];
			// The a/b pair's symmetric energy is unchanged by a swap.
			if ( an !== b && an !== a ) {

				const value = shifts[ an ];
				delta += weights[ k ] * ( potential[ Math.round( Math.abs( bv - value ) * resolution ) ] - potential[ Math.round( Math.abs( av - value ) * resolution ) ] );

			}

			if ( bn !== a && bn !== b ) {

				const value = shifts[ bn ];
				delta += weights[ k ] * ( potential[ Math.round( Math.abs( av - value ) * resolution ) ] - potential[ Math.round( Math.abs( bv - value ) * resolution ) ] );

			}

		}

		if ( delta < 0 ) {

			shifts[ a ] = bv;
			shifts[ b ] = av;

		}

	}

	return shifts;

}

/** Continuous Weyl progression; frame is never wrapped at the tile depth. */
export function sample( tile, x, y, frame ) {

	const size = Math.sqrt( tile.length );
	const mask = size - 1;
	const phase = tile[ ( y & mask ) * size + ( x & mask ) ];
	const rotation = ( Math.imul( frame, 2654435769 ) >>> 8 ) / 16777216;
	const value = Math.fround( phase + rotation );
	return value - Math.floor( value );

}

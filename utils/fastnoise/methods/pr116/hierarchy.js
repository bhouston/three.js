// Copied from bhouston/three.js PR116 at 54a6dfe336cd33fc1fe129e4e1e6b10900395832; formatting only.
// Hierarchical spatial strata with a digitally shifted temporal van der Corput sequence.
// Inspired by Ahmed and Wonka's hierarchical sampling, not an implementation of ZSampler.

function hash( value ) {

	value = Math.imul( value ^ ( value >>> 16 ), 0x7feb352d );
	value = Math.imul( value ^ ( value >>> 15 ), 0x846ca68b );
	return ( value ^ ( value >>> 16 ) ) >>> 0;

}

function reverse24( value ) {

	value = ( ( value >>> 1 ) & 0x55555555 ) | ( ( value & 0x55555555 ) << 1 );
	value = ( ( value >>> 2 ) & 0x33333333 ) | ( ( value & 0x33333333 ) << 2 );
	value = ( ( value >>> 4 ) & 0x0f0f0f0f ) | ( ( value & 0x0f0f0f0f ) << 4 );
	value = ( ( value >>> 8 ) & 0x00ff00ff ) | ( ( value & 0x00ff00ff ) << 8 );
	return ( ( value >>> 16 ) | ( value << 16 ) ) >>> 8;

}

export function generate( { size = 128, depth = 32, seed = 1 } = {} ) {

	if ( ! Number.isInteger( size ) || size < 1 || size > 4096 || ! Number.isInteger( depth ) || depth < 1 || depth > 16777216 ) {

		throw new RangeError( 'size must be an integer in [1,4096]; depth in [1,16777216].' );

	}

	const area = size * size;
	const result = new Float32Array( area * depth );
	const ranks = new Uint32Array( area );
	const levels = Math.ceil( Math.log2( size ) );
	const salt = hash( seed | 0 );

	for ( let y = 0; y < size; y ++ ) {

		for ( let x = 0; x < size; x ++ ) {

			let rank = hash( Math.imul( y, size ) + x ^ salt ) & 0xffffff;

			for ( let level = 0; level < levels; level ++ ) {

				// Each sibling quartet receives all four leading base-four digits.
				const node = hash( Math.imul( x >>> ( level + 1 ), 0x1f123bb5 ) ^ Math.imul( y >>> ( level + 1 ), 0x5f356495 ) ^ Math.imul( level + 1, 0x6c8e9cf5 ) ^ salt );
				let digit = ( ( x >>> level ) & 1 ) | ( ( ( y >>> level ) & 1 ) << 1 );
				// Eight square symmetries preserve a permutation without a lookup table.
				if ( node & 4 ) digit = ( digit >>> 1 ) | ( ( digit & 1 ) << 1 );
				digit ^= node & 3;
				const shift = 22 - 2 * level;
				rank = ( rank & ~ ( 3 << shift ) ) | ( digit << shift );

			}

			ranks[ y * size + x ] = rank;

		}

	}

	for ( let z = 0; z < depth; z ++ ) {

		const temporal = reverse24( z );
		const offset = z * area;
		for ( let i = 0; i < area; i ++ ) result[ offset + i ] = ( ranks[ i ] ^ temporal ) / 16777216;

	}

	return result;

}

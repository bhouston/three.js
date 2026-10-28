// Arithmetic scalar sampling candidates; these are not optimized blue-noise masks.
const f32 = Math.fround;

function fract( value ) {

	return value - Math.floor( value );

}

function hash( value ) {

	value = Math.imul( value ^ ( value >>> 16 ), 0x21f0aaad );
	value = Math.imul( value ^ ( value >>> 15 ), 0x735a2d97 );
	return ( value ^ ( value >>> 15 ) ) >>> 0;

}

function pixelHash( x, y, seed ) {

	return hash( Math.imul( x, 0x9e3779b1 ) ^ Math.imul( y, 0x85ebca77 ) ^ seed );

}

export function animatedIGN( x, y, frame, seed = 1 ) {

	const shift = f32( frame * f32( 5.588238 ) );
	const offset = f32( ( hash( seed ) >>> 8 ) / 16777216 * 256 );
	const px = f32( f32( x + shift ) + offset ), py = f32( f32( y + shift ) + offset );
	return fract( f32( f32( 52.9829189 ) * fract( f32( f32( f32( .06711056 ) * px ) + f32( f32( .00583715 ) * py ) ) ) ) );

}

// Every aligned 2^k square contains all 4^k leading ranks. Each parent
// independently permutes its four children using an affine permutation.
function spatialRank( x, y, seed, levels ) {

	let rank = 0;
	for ( let level = 0; level < levels; level ++ ) {

		const scramble = pixelHash( x >> ( level + 1 ), y >> ( level + 1 ), seed ^ Math.imul( level + 1, 0x632be5ab ) );
		let a = ( x >> level ) & 1;
		let b = ( y >> level ) & 1;
		if ( scramble & 4 ) [ a, b ] = [ b, a ];
		const child = ( ( a ^ b ) << 1 ) | b;
		rank |= ( child ^ ( scramble & 3 ) ) << ( 30 - 2 * level );

	}

	return rank >>> 0;

}

export function tiledBayerWeyl( x, y, frame, seed = 1 ) {

	const rank = spatialRank( x, y, seed, 3 );
	const tileShift = pixelHash( x >> 3, y >> 3, seed ^ 0x02e5be93 );
	return ( ( rank + tileShift + Math.imul( frame, 2654435769 ) ) >>> 8 ) / 16777216;

}


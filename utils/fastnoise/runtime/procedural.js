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

function reverseBits( value ) {

	value = ( ( value >>> 1 ) & 0x55555555 ) | ( ( value & 0x55555555 ) << 1 );
	value = ( ( value >>> 2 ) & 0x33333333 ) | ( ( value & 0x33333333 ) << 2 );
	value = ( ( value >>> 4 ) & 0x0f0f0f0f ) | ( ( value & 0x0f0f0f0f ) << 4 );
	value = ( ( value >>> 8 ) & 0x00ff00ff ) | ( ( value & 0x00ff00ff ) << 8 );
	return ( ( value >>> 16 ) | ( value << 16 ) ) >>> 0;

}

export function animatedIGN( x, y, frame, seed = 1 ) {

	const shift = f32( frame * f32( 5.588238 ) );
	const offset = f32( ( hash( seed ) >>> 8 ) / 16777216 * 256 );
	const px = f32( f32( x + shift ) + offset ), py = f32( f32( y + shift ) + offset );
	return fract( f32( f32( 52.9829189 ) * fract( f32( f32( f32( .06711056 ) * px ) + f32( f32( .00583715 ) * py ) ) ) ) );

}

export function r2Weyl( x, y, frame, seed = 1 ) {

	return ( ( Math.imul( x, 3242174889 ) + Math.imul( y, 2447445414 ) + Math.imul( frame, 2654435769 ) + hash( seed ) ) >>> 8 ) / 16777216;

}

export function hashedDigital( x, y, frame, seed = 1 ) {

	return ( ( reverseBits( frame ) ^ pixelHash( x, y, seed ) ) >>> 8 ) / 16777216;

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

export function hierarchicalDigital( x, y, frame, seed = 1 ) {

	const spatial = spatialRank( x, y, seed, 8 );
	const jitter = pixelHash( x, y, seed ^ 0x68bc21eb ) & 0xffff;
	return ( ( spatial ^ jitter ^ reverseBits( frame ) ) >>> 8 ) / 16777216;

}

export function tiledBayerWeyl( x, y, frame, seed = 1 ) {

	const rank = spatialRank( x, y, seed, 3 );
	const tileShift = pixelHash( x >> 3, y >> 3, seed ^ 0x02e5be93 );
	return ( ( rank + tileShift + Math.imul( frame, 2654435769 ) ) >>> 8 ) / 16777216;

}

export const proceduralSamplers = {
	white: ( x, y, frame, seed = 1 ) => ( pixelHash( x, y, seed ^ hash( frame ) ) >>> 8 ) / 16777216,
	'animated-ign': animatedIGN,
	'r2-weyl': r2Weyl,
	'hashed-digital': hashedDigital,
	'hierarchical-digital': hierarchicalDigital,
	'tiled-bayer-weyl': tiledBayerWeyl
};

const wgslHelpers = /* wgsl */`
fn hash(v0:u32)->u32 { var v=v0; v=(v^(v>>16u))*0x21f0aaadu; v=(v^(v>>15u))*0x735a2d97u; return v^(v>>15u); }
fn pixelHash(x:u32,y:u32,s:u32)->u32 { return hash((x*0x9e3779b1u) ^ (y*0x85ebca77u) ^ s); }
fn unit(v:u32)->f32 { return f32(v>>8u)/16777216.0; }
fn spatialRank(x:u32,y:u32,s:u32,levels:u32)->u32 {
 var rank=0u;
 for(var l=0u;l<levels;l++) {
  let q=pixelHash(x>>(l+1u),y>>(l+1u),s^((l+1u)*0x632be5abu));
  var a=(x>>l)&1u; var b=(y>>l)&1u;
  if((q&4u)!=0u) { let t=a; a=b; b=t; }
  rank|=((((a^b)<<1u)|b)^(q&3u))<<(30u-2u*l);
 }
 return rank;
}
`;

const wgslBodies = {
	white: 'return unit(pixelHash(x,y,seed^hash(frame)));',
	'animated-ign': 'let shift=f32(frame)*5.588238; let offset=unit(hash(seed))*256.0; return fract(52.9829189*fract(0.06711056*(f32(x)+shift+offset)+0.00583715*(f32(y)+shift+offset)));',
	'r2-weyl': 'return unit(x*3242174889u+y*2447445414u+frame*2654435769u+hash(seed));',
	'hashed-digital': 'return unit(reverseBits(frame)^pixelHash(x,y,seed));',
	'hierarchical-digital': 'return unit(spatialRank(x,y,seed,8u)^(pixelHash(x,y,seed^0x68bc21ebu)&0xffffu)^reverseBits(frame));',
	'tiled-bayer-weyl': 'return unit(spatialRank(x,y,seed,3u)+pixelHash(x>>3u,y>>3u,seed^0x02e5be93u)+frame*2654435769u);'
};

export const proceduralWGSL = Object.fromEntries( Object.entries( wgslBodies ).map( ( [ name, body ] ) => [ name, `${wgslHelpers}\nfn sample(x:u32,y:u32,frame:u32,seed:u32)->f32 { ${body} }` ] ) );

export const samplers = proceduralSamplers;
export const wgslSamplers = proceduralWGSL;

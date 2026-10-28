// Same quantization, wrapped blur, fetch convention and scalar metrics as compare.py.
// Also evaluate discontinuous integrands; good raw-value spectra alone are insufficient.
import fs from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';

const dir = path.resolve( process.argv[ 2 ] || 'noise-results' );
const S = 128, Z = 32, N = S * S;
const kernel = Array.from( { length: 7 }, ( _, i ) => Math.exp( - 0.5 * ( i - 3 ) ** 2 ) );
const total = kernel.reduce( ( a, b ) => a + b );
const weights = kernel.map( v => v / total );
function blur( a ) {

	let src = a;
	for ( const axis of [ 0, 1 ] ) {

		const dst = new Float64Array( N );
		for ( let y = 0; y < S; y ++ ) for ( let x = 0; x < S; x ++ ) for ( let k = - 3; k <= 3; k ++ ) dst[ y * S + x ] += weights[ k + 3 ] * src[ ( ( y + ( axis ? k : 0 ) + S ) % S ) * S + ( x + ( axis ? 0 : k ) + S ) % S ];
		src = dst;

	}

	return src;

}

const rmse = ( a, target ) => Math.sqrt( a.reduce( ( sum, v ) => sum + ( v - target ) ** 2, 0 ) / a.length );
const std = a => rmse( a, a.reduce( ( sum, v ) => sum + v, 0 ) / a.length );
function fft( re, im ) {

	const n = re.length;
	for ( let i = 1, j = 0; i < n; i ++ ) {

		let bit = n >> 1;
		for ( ; j & bit; bit >>= 1 ) j ^= bit;
		j ^= bit;
		if ( i < j ) {

			[ re[ i ], re[ j ] ] = [ re[ j ], re[ i ] ]; [ im[ i ], im[ j ] ] = [ im[ j ], im[ i ] ];

		}

	}

	for ( let len = 2; len <= n; len *= 2 ) for ( let i = 0; i < n; i += len ) for ( let j = 0; j < len / 2; j ++ ) {

		const angle = - 2 * Math.PI * j / len, c = Math.cos( angle ), s = Math.sin( angle ), a = i + j, b = a + len / 2;
		const r = re[ b ] * c - im[ b ] * s, v = re[ b ] * s + im[ b ] * c;
		re[ b ] = re[ a ] - r; im[ b ] = im[ a ] - v; re[ a ] += r; im[ a ] += v;

	}

}

function spectra( v ) {

	const tp = new Float64Array( Z ), sp = new Float64Array( N );
	for ( let p = 0; p < N; p ++ ) {

		const re = Float64Array.from( { length: Z }, ( _, z ) => v[ z * N + p ] - 0.5 ), im = new Float64Array( Z ); fft( re, im );
		for ( let z = 0; z < Z; z ++ ) tp[ z ] += re[ z ] ** 2 + im[ z ] ** 2;

	}

	for ( let z = 0; z < Z; z ++ ) {

		const re = v.slice( z * N, ( z + 1 ) * N ).map( x => x - 0.5 ), im = new Float64Array( N );
		for ( let y = 0; y < S; y ++ ) fft( re.subarray( y * S, ( y + 1 ) * S ), im.subarray( y * S, ( y + 1 ) * S ) );
		for ( let x = 0; x < S; x ++ ) {

			const r = Float64Array.from( { length: S }, ( _, y ) => re[ y * S + x ] ), m = Float64Array.from( { length: S }, ( _, y ) => im[ y * S + x ] ); fft( r, m );
			for ( let y = 0; y < S; y ++ ) sp[ y * S + x ] += r[ y ] ** 2 + m[ y ] ** 2;

		}

	}

	let low = 0, high = 0, nl = 0, nh = 0;
	for ( let y = 0; y < S; y ++ ) for ( let x = 0; x < S; x ++ ) {

		const r = Math.hypot( Math.min( x, S - x ) / S, Math.min( y, S - y ) / S );
		if ( r > 0 && r < 0.1 ) {

			low += sp[ y * S + x ]; nl ++;

		}

		if ( r > 0.3 ) {

			high += sp[ y * S + x ]; nh ++;

		}

	}

	return { tmpLow: ( tp[ 1 ] + tp[ 2 ] + tp[ 3 ] ) / 3 / ( tp.slice( 1 ).reduce( ( a, b ) => a + b ) / 31 ), spLow: low / nl / ( high / nh ) };

}

function filtered( v, target = 0.5 ) {

	const acc = new Float64Array( N ), result = {};
	for ( let n = 1; n <= 256; n ++ ) {

		const frame = n - 1, cycle = Math.floor( frame / Z );
		for ( let y = 0; y < S; y ++ ) for ( let x = 0; x < S; x ++ ) acc[ y * S + x ] += v[ ( frame % Z ) * N + ( ( y + 71 * cycle ) % S ) * S + ( x + 37 * cycle ) % S ];
		if ( [ 1, 4, 32, 256 ].includes( n ) ) {

			const avg = acc.map( x => x / n );
			result[ 'pp' + n ] = std( avg ); result[ 'bl' + n ] = std( blur( avg ) ); result[ 'rmse' + n ] = rmse( blur( avg ), target );

		}

	}

	const h = new Float64Array( N );
	for ( let z = 0; z < Z; z ++ ) for ( let p = 0; p < N; p ++ ) h[ p ] += v[ z * N + p ] / Z;
	let ema = 0, emab = 0;
	for ( let it = 0; it < 4; it ++ ) for ( let z = 0; z < Z; z ++ ) {

		for ( let p = 0; p < N; p ++ ) h[ p ] += 0.1 * ( v[ z * N + p ] - h[ p ] );
		if ( it === 3 ) {

			ema += std( h ) / Z; emab += rmse( blur( h ), target ) / Z;

		}

	}

	return { ...result, ema, 'ema+bl': emab };

}

const rows = [];
for ( const file of fs.readdirSync( dir ).filter( f => f.endsWith( '.f32' ) ) ) {

	const bytes = fs.readFileSync( path.join( dir, file ) );
	const raw = new Float32Array( bytes.buffer, bytes.byteOffset, bytes.byteLength / 4 );
	if ( raw.length !== N * Z || raw.some( x => ! Number.isFinite( x ) || x < 0 || x >= 1 ) ) throw new Error( 'Invalid volume: ' + file );
	const v = Float64Array.from( raw, x => ( Math.floor( x * 256 ) + 0.5 ) / 256 );
	const row = { name: file.replace( '.f32', '' ), raw: { ...spectra( v ), ...filtered( v ) }, thresholds: {} };
	for ( const threshold of [ 0.1, 0.25, 0.5, 0.75, 0.9 ] ) row.thresholds[ threshold ] = filtered( v.map( x => Number( x < threshold ) ), threshold );
	rows.push( row ); console.log( row.name, row.raw );
	const png = new PNG( { width: S * 4, height: S } );
	const panel = blur( v.subarray( 0, N ).map( x => Number( x < 0.5 ) ) );
	const mean = new Float64Array( N );
	for ( let z = 0; z < Z; z ++ ) for ( let p = 0; p < N; p ++ ) mean[ p ] += v[ z * N + p ] / Z;
	for ( let y = 0; y < S; y ++ ) for ( let x = 0; x < S * 4; x ++ ) {

		const px = x % S, p = y * S + px, which = Math.floor( x / S );
		const value = which === 0 ? v[ p ] : which === 1 ? Number( v[ p ] < 0.5 ) : which === 2 ? 0.5 + 4 * ( panel[ p ] - 0.5 ) : 0.5 + 8 * ( mean[ p ] - 0.5 );
		const o = ( y * S * 4 + x ) * 4;
		png.data[ o ] = png.data[ o + 1 ] = png.data[ o + 2 ] = Math.max( 0, Math.min( 255, Math.round( value * 255 ) ) ); png.data[ o + 3 ] = 255;

	}

	fs.writeFileSync( path.join( dir, row.name + '.png' ), PNG.sync.write( png ) );

}

fs.writeFileSync( path.join( dir, 'quality.json' ), JSON.stringify( rows, null, 2 ) );

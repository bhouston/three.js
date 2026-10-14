// Original synthetic pixels: CC0-1.0. Produces inputs for avifenc and cjxl.
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { deflateSync } from 'node:zlib';

const output = path.resolve( process.argv[ 2 ] || 'examples/textures/hdr-native' );
await mkdir( output, { recursive: true } );
const width = 160, height = 32;
const nits = [ 0, 50, 203, 406, 812, 1624, 4000, 10000 ];
const m1 = 2610 / 16384, m2 = 2523 / 32;
const c1 = 3424 / 4096, c2 = 2413 / 128, c3 = 2392 / 128;
const codes = nits.map( value => {

	const p = Math.pow( value / 10000, m1 );
	return Math.round( Math.pow( ( c1 + c2 * p ) / ( 1 + c3 * p ), m2 ) * 4095 );

} );
const codes16 = codes.map( value => Math.round( value * 65535 / 4095 ) );
const raw = Buffer.alloc( height * ( 1 + width * 6 ) );
const ppm = Buffer.alloc( width * height * 6 );
for ( let y = 0; y < height; y ++ ) {

	for ( let x = 0; x < width; x ++ ) {

		for ( let channel = 0; channel < 3; channel ++ ) {

			const value = codes16[ Math.floor( x / 20 ) ];
			raw.writeUInt16BE( value, y * ( 1 + width * 6 ) + 1 + x * 6 + channel * 2 );
			ppm.writeUInt16BE( value, ( y * width + x ) * 6 + channel * 2 );

		}

	}

}

function chunk( name, data ) {

	const bytes = Buffer.concat( [ Buffer.from( name ), data ] );
	let crc = 0xffffffff;
	for ( const byte of bytes ) {

		crc ^= byte;
		for ( let i = 0; i < 8; i ++ ) crc = ( crc >>> 1 ) ^ ( ( crc & 1 ) ? 0xedb88320 : 0 );

	}

	const length = Buffer.alloc( 4 ), checksum = Buffer.alloc( 4 );
	length.writeUInt32BE( data.length );
	checksum.writeUInt32BE( ( crc ^ 0xffffffff ) >>> 0 );
	return Buffer.concat( [ length, bytes, checksum ] );

}

const header = Buffer.alloc( 13 );
header.writeUInt32BE( width );
header.writeUInt32BE( height, 4 );
header[ 8 ] = 16;
header[ 9 ] = 2;
await writeFile( path.join( output, 'neutral-pq.png' ), Buffer.concat( [
	Buffer.from( [ 137, 80, 78, 71, 13, 10, 26, 10 ] ),
	chunk( 'IHDR', header ), chunk( 'IDAT', deflateSync( raw ) ), chunk( 'IEND', Buffer.alloc( 0 ) )
] ) );
await writeFile( path.join( output, 'neutral-pq.ppm' ), Buffer.concat( [ Buffer.from( `P6\n${width} ${height}\n65535\n` ), ppm ] ) );
const patches = codes.map( ( code12, i ) => {

	const p = Math.pow( code12 / 4095, 1 / m2 );
	const luminance = 10000 * Math.pow( Math.max( p - c1, 0 ) / ( c2 - c3 * p ), 1 / m1 );
	return { x: i * 20 + 10, y: 16, code12, nits: luminance, linear: luminance / 203 };

} );
await writeFile( path.join( output, 'neutral-pq.json' ), JSON.stringify( {
	width, height, primaries: 'BT.2020', transfer: 'PQ', referenceWhiteNits: 203,
	patches, jxlCode16: codes16, license: 'CC0-1.0 (generated synthetic fixtures)'
}, null, 2 ) + '\n' );

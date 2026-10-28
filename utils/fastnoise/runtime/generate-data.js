// Evaluation streams only; applications call the arithmetic samplers directly.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { PNG } from 'pngjs';
import { samplers } from './procedural.js';
import { generate, sample } from './phase-tile.js';

const out = process.argv[ 2 ];
if ( ! out ) throw new Error( 'Usage: node runtime/generate-data.js OUTPUT_DIR' );
fs.mkdirSync( out, { recursive: true } );
const size = 128, frames = 159, count = size * size;
const atlasFile = new URL( '../../../examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png', import.meta.url );
const atlas = PNG.sync.read( fs.readFileSync( atlasFile ) );
const report = { size, frames, seeds: [ 1, 2 ], atlasBytes: fs.statSync( atlasFile ).size, methods: [] };
for ( const seed of report.seeds ) {

	const profiles = Object.entries( samplers ).map( ( [ name, fn ] ) => ( { name, fn, persistentBytes: 0, startupMilliseconds: 0 } ) );
	for ( const [ tileSize, passes ] of [[ 16, 8 ], [ 32, 0 ], [ 32, 2 ], [ 32, 8 ], [ 64, 8 ], [ 128, 8 ]] ) {

		const start = performance.now(), tile = generate( { size: tileSize, passes, seed } );
		profiles.push( { name: `phase-${tileSize}-p${passes}`, fn: ( x, y, n ) => sample( tile, x, y, n ), persistentBytes: tile.byteLength, startupMilliseconds: performance.now() - start } );

	}

	profiles.push( { name: 'FAST-scalar', persistentBytes: 128 * 128 * 32, startupMilliseconds: null, fn: ( x, y, n ) => {

		const cycle = Math.floor( n / 32 );
		return ( atlas.data[ ( ( n % 32 * 128 + ( y + 71 * cycle ) % 128 ) * atlas.width + ( x + 37 * cycle ) % 128 ) * 4 + 2 ] + .5 ) / 256;

	} } );
	for ( const { fn, ...profile } of profiles ) {

		const data = new Float32Array( frames * count );
		for ( let n = 0; n < frames; n ++ ) for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

			const value = fn( x, y, n, seed );
			if ( ! Number.isFinite( value ) || value < 0 || value >= 1 ) throw new Error( `Invalid sample: ${profile.name}` );
			data[ n * count + y * size + x ] = value;

		}

		const file = `${profile.name}-seed${seed}.f32`, bytes = Buffer.from( data.buffer );
		fs.writeFileSync( path.join( out, file ), bytes );
		report.methods.push( { ...profile, seed, file, sha256: createHash( 'sha256' ).update( bytes ).digest( 'hex' ) } );
		console.log( file );

	}

}

fs.writeFileSync( path.join( out, 'streams.json' ), JSON.stringify( report, null, 2 ) + '\n' );

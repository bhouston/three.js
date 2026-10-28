// Export seed-1 scalar volumes for a common integrand test. The reference
// directory must contain hierarchy.mjs, lattice.mjs and relaxation.mjs from
// PR #116 commit 54a6dfe336cd33fc1fe129e4e1e6b10900395832.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { PNG } from 'pngjs';
import { generate as filterRank } from './methods/filter-rank.js';
import { generate as hierarchy } from './methods/hierarchy.js';
import { generate as localEnergy } from './methods/local-energy.js';

const [ referenceDirectory, outputDirectory ] = process.argv.slice( 2 );
if ( ! referenceDirectory || ! outputDirectory ) throw new Error( 'Usage: node compare-pr116.js REFERENCE_MODULE_DIRECTORY OUTPUT_DIRECTORY' );
fs.mkdirSync( outputDirectory, { recursive: true } );
const save = ( name, volume ) => fs.writeFileSync( path.join( outputDirectory, name + '.f32' ), Buffer.from( volume.buffer, volume.byteOffset, volume.byteLength ) );
for ( const [ name, generate ] of [[ '117-filter-rank', filterRank ], [ '117-hierarchy', hierarchy ], [ '117-threshold-energy', localEnergy ]] ) save( name, generate( { seed: 1 } ) );
for ( const name of [ 'hierarchy', 'lattice', 'relaxation' ] ) {

	const { generate } = await import( pathToFileURL( path.resolve( referenceDirectory, name + '.mjs' ) ).href );
	save( '116-' + name, generate( { seed: 1 } ) );

}

const image = PNG.sync.read( fs.readFileSync( new URL( '../../examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png', import.meta.url ) ) );
for ( const [ channel, name ] of [[ 0, 'FAST-vector-R' ], [ 2, 'FAST-scalar-B' ]] ) {

	const volume = Float32Array.from( { length: 32 * 128 * 128 }, ( _, i ) => {

		const z = Math.floor( i / 16384 ), p = i % 16384;
		return ( image.data[ ( ( z * 128 + Math.floor( p / 128 ) ) * 256 + p % 128 ) * 4 + channel ] + 0.5 ) / 256;

	} );
	save( name, volume );

}

save( 'white', filterRank( { seed: 1, iterations: 0 } ) );

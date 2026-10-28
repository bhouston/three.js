// Import published quantized volumes for the common scorer, not a new GPU run.
// node utils/fastnoise/experiments/import-pr117.js RAW_DIRECTORY
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { PNG } from 'pngjs';

const pin = '2a065526980261bcf5b129f07ddb12ff7eda1f91';
const root = fileURLToPath( new URL( '../../../', import.meta.url ) );
const directory = path.resolve( process.argv[ 2 ] );
const prefix = 'examples/textures/noises/experimental/';
const names = [ 'fast', 'light-seam-03', 'seam-polish', 'pareto-polish', 'continuation-deep', 'continuation-time-control' ];
const sha = bytes => createHash( 'sha256' ).update( bytes ).digest( 'hex' );
function read( file ) {

	if ( ! /^[\w/.-]+$/.test( file ) ) throw new Error( 'Unsafe source path.' );
	const windows = process.platform === 'win32';
	const run = spawnSync( windows ? 'cmd.exe' : 'git-dedup', windows ? [ '/d', '/s', '/c', `git-dedup.cmd show ${pin}:${file}` ] : [ 'show', `${pin}:${file}` ], { cwd: root, maxBuffer: 32 * 1024 * 1024 } );
	if ( run.status !== 0 ) throw new Error( 'Fetch the pinned PR117 source using git-dedup first: ' + run.stderr );
	return run.stdout;

}

const source = JSON.parse( read( prefix + 'manifest.json' ) );
const generation = JSON.parse( read( 'utils/fastnoise/results/frontier-generation.json' ) );
const provenance = { commit: pin, seed: 1, note: 'Published PNGs, not newly generated samples. Midpoint Float32 decode is exact for this quantize-first scorer. Original Float32 and decoded Float32 hashes differ.', profiles: [] };
fs.mkdirSync( directory, { recursive: true } );
for ( const name of names ) {

	const entry = source.entries.find( e => e.id === name );
	if ( ! entry ) throw new Error( 'Missing profile ' + name );
	const bytes = read( prefix + entry.file );
	if ( sha( bytes ) !== entry.pngSHA256 ) throw new Error( 'PNG hash mismatch: ' + name );
	const image = PNG.sync.read( bytes );
	if ( image.width !== 128 || image.height !== 4096 ) throw new Error( 'Invalid dimensions.' );
	const values = new Float32Array( 128 * 128 * 32 );
	const quantized = new Uint8Array( values.length );
	for ( let i = 0; i < values.length; i ++ ) {

		const offset = i * 4;
		if ( image.data[ offset ] !== image.data[ offset + 1 ] || image.data[ offset ] !== image.data[ offset + 2 ] || image.data[ offset + 3 ] !== 255 ) throw new Error( 'Expected opaque grayscale.' );
		quantized[ i ] = image.data[ offset ];
		values[ i ] = ( quantized[ i ] + .5 ) / 256;

	}

	const raw = Buffer.from( values.buffer );
	fs.writeFileSync( path.join( directory, 'pr117-' + name + '-s1.f32' ), raw );
	fs.writeFileSync( path.join( directory, entry.file ), bytes );
	provenance.profiles.push( { name: 'pr117-' + name + '-s1', source: entry, quantizedSHA256: sha( quantized ), decodedSHA256: sha( raw ), generation: generation.runs.find( r => r.name === name + '-seed1' ) } );
	console.log( 'Imported', name );

}

fs.writeFileSync( path.join( root, 'utils/fastnoise/results/frontier/pr117-import.json' ), JSON.stringify( provenance, null, 2 ) + '\n' );

// Offline only. The example never imports or runs any of these constructors.
// Usage: node generate-example-noises.js RAW_DIRECTORY [--resume]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const out = path.resolve( process.argv[ 2 ] || 'noise-example-data' );
const resume = process.argv.includes( '--resume' );
const root = path.dirname( fileURLToPath( import.meta.url ) );
fs.mkdirSync( out, { recursive: true } );
const read = name => JSON.parse( fs.readFileSync( path.join( root, name ) ) );
const baseline = read( 'research-baseline.json' );
const screen = read( 'research-screen.json' );
const followup = read( 'research-followup.json' );
const confirm = read( 'research-confirm.json' );
const profiles = [ ...baseline, ...screen.filter( p => p.kind === 'gpu' && ! p.name.startsWith( 'control-' ) ), ...followup.filter( p => p.kind === 'gpu' ), ...confirm.filter( p => p.name === 'continuation-time-control' ) ].map( p => ( { ...p, seeds: [ 1 ] } ) );
const profileFile = path.join( out, 'example-gpu-profiles.json' );
fs.writeFileSync( profileFile, JSON.stringify( profiles, null, 2 ) + '\n' );
if ( resume && fs.existsSync( path.join( out, 'generation.json' ) ) ) {

	const previous = JSON.parse( fs.readFileSync( path.join( out, 'generation.json' ) ) );
	for ( const run of previous.runs ) if ( ! fs.existsSync( path.join( out, run.name + '.f32' ) ) ) throw new Error( 'Cannot resume with a missing completed volume: ' + run.name );

}

const gpu = spawnSync( process.execPath, [ path.join( root, 'research-run.js' ), profileFile, out, ...resume ? [ '--resume' ] : [] ], { stdio: 'inherit' } );
if ( gpu.error ) throw gpu.error;
if ( gpu.status !== 0 ) throw new Error( 'Offline GPU generation failed; no software fallback is permitted.' );

const cpuProfiles = [
	{ name: 'filter-rank', module: './methods/filter-rank.js', options: {} },
	{ name: 'hierarchy', module: './methods/hierarchy.js', options: {} },
	{ name: 'local-energy', module: './methods/local-energy.js', options: {} },
	...[ 'hierarchy', 'lattice', 'relaxation' ].map( name => ( { name: 'pr116-' + name, module: `./methods/pr116/${name}.js`, options: {} } ) ),
	...screen.filter( p => p.kind === 'cpu' ),
	...followup.filter( p => p.kind === 'cpu' )
];
for ( const profile of cpuProfiles ) {

	const file = path.join( out, profile.name + '-seed1.f32' );
	if ( resume && fs.existsSync( file ) ) continue;
	const { generate } = await import( new URL( profile.module, import.meta.url ) );
	const options = { ...profile.options, seed: 1 };
	if ( profile.initial ) {

		const bytes = fs.readFileSync( path.join( out, profile.initial.replace( '{seed}', 1 ) + '.f32' ) );
		options.initialVolume = new Float32Array( bytes.buffer.slice( bytes.byteOffset, bytes.byteOffset + bytes.byteLength ) );

	}

	const volume = generate( options );
	fs.writeFileSync( file, Buffer.from( volume.buffer, volume.byteOffset, volume.byteLength ) );
	console.log( 'Generated', profile.name );

}

// Bind the example assets to the actual volumes used in the earlier reports.
const frontier = read( 'results/frontier.json' ).methods;
const comparison = read( 'results/pr116-comparison.json' ).methods;
const initialNames = { 'filter-rank': '117-filter-rank', 'hierarchy': '117-hierarchy', 'local-energy': '117-threshold-energy', 'pr116-hierarchy': '116-hierarchy', 'pr116-lattice': '116-lattice', 'pr116-relaxation': '116-relaxation' };
const verified = [];
for ( const file of fs.readdirSync( out ).filter( f => f.endsWith( '.f32' ) ) ) {

	const name = file.slice( 0, - 4 ), id = name.replace( /-seed1$/, '' );
	const hash = createHash( 'sha256' ).update( fs.readFileSync( path.join( out, file ) ) ).digest( 'hex' );
	const expected = frontier[ name ]?.sha256 ?? comparison[ initialNames[ id ] ]?.sha256_f32;
	if ( ! expected || hash !== expected ) throw new Error( 'Generated volume differs from the recorded research: ' + name );
	verified.push( { name, sha256: hash } );

}

fs.writeFileSync( path.join( out, 'verified-volumes.json' ), JSON.stringify( verified, null, 2 ) + '\n' );
console.log( 'Verified', verified.length, 'volumes against committed research hashes.' );

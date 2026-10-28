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

const profiles = [ ...baseline, ...screen.filter( p => p.name === 'continuation-polish' ), ...followup.filter( p => p.name === 'light-seam-03' ) ].map( p => ( { ...p, seeds: [ 1 ] } ) );
const profileFile = path.join( out, 'example-gpu-profiles.json' );
fs.writeFileSync( profileFile, JSON.stringify( profiles, null, 2 ) + '\n' );
if ( resume && fs.existsSync( path.join( out, 'generation.json' ) ) ) {

	const previous = JSON.parse( fs.readFileSync( path.join( out, 'generation.json' ) ) );
	for ( const run of previous.runs ) if ( ! fs.existsSync( path.join( out, run.name + '.f32' ) ) ) throw new Error( 'Cannot resume with a missing completed volume: ' + run.name );

}

const gpu = spawnSync( process.execPath, [ path.join( root, 'research-run.js' ), profileFile, out, ...resume ? [ '--resume' ] : [] ], { stdio: 'inherit' } );
if ( gpu.error ) throw gpu.error;
if ( gpu.status !== 0 ) throw new Error( 'Offline GPU generation failed; no software fallback is permitted.' );

// The pinned runner reproduces all three frozen PR116 profiles on both seeds.
// Only its annealed seed-1 result is shipped by the example.
const pr116 = spawnSync( process.execPath, [ path.join( root, 'pr116-frontier/run.js' ), out, ...resume ? [ '--resume' ] : [] ], { stdio: 'inherit' } );
if ( pr116.error ) throw pr116.error;
if ( pr116.status !== 0 ) throw new Error( 'Offline PR116 generation failed; no software fallback is permitted.' );

// Bind the example assets to the actual volumes used in the earlier reports.
const frontier = read( 'results/frontier.json' ).methods;
const verified = [];
for ( const file of [ 'light-seam-03-seed1.f32', 'pr116-annealed-seed1.f32' ] ) {

	const name = file.slice( 0, - 4 );
	const hash = createHash( 'sha256' ).update( fs.readFileSync( path.join( out, file ) ) ).digest( 'hex' );
	const expected = frontier[ name ]?.sha256;
	if ( ! expected || hash !== expected ) throw new Error( 'Generated volume differs from the recorded research: ' + name );
	verified.push( { name, sha256: hash } );

}

fs.writeFileSync( path.join( out, 'verified-volumes.json' ), JSON.stringify( verified, null, 2 ) + '\n' );
console.log( 'Verified', verified.length, 'volumes against committed research hashes.' );

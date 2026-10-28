// Preserve prior generation records while adding the pinned PR116 reproduction.
import assert from 'node:assert/strict';
import fs from 'node:fs';

const [ input, output = 'utils/fastnoise/results/frontier-generation.json' ] = process.argv.slice( 2 );
if ( ! input ) throw new Error( 'Usage: node merge-pr116-frontier.js PR116_GENERATION_JSON [COMBINED_JSON]' );
const added = JSON.parse( fs.readFileSync( input ) );
const combined = JSON.parse( fs.readFileSync( output ) );
assert.equal( added.sourceCommit, 'd8b1acffaada09223b5133d965f3a7cf9028fe5b' );
assert.equal( added.environment.gpu.fallback, false );
const names = [ 'annealed', 'greedy', 'autocorrelation' ].flatMap( name => [ 1, 2 ].map( seed => `pr116-${name}-seed${seed}` ) );
assert.deepEqual( added.runs.map( row => row.name ).sort(), names.slice().sort() );
for ( const row of added.runs ) {

	assert.ok( Number.isFinite( row.secondsPerVolume ) && row.secondsPerVolume > 0 );
	assert.match( row.sha256, /^[0-9a-f]{64}$/ );
	const next = { ...row, totalSecondsPerVolume: row.secondsPerVolume, sourceCommit: added.sourceCommit };
	const index = combined.runs.findIndex( item => item.name === row.name );
	if ( index < 0 ) combined.runs.push( next );
	else combined.runs[ index ] = next;

}

combined.pr116Reproduction = { sourceCommit: added.sourceCommit, sourceSHA256: added.sourceSHA256, environment: added.environment, adaptations: added.adaptations };
fs.writeFileSync( output, JSON.stringify( combined, null, 2 ) + '\n' );
console.log( 'Merged six scratch profiles; prior construction records retained.' );

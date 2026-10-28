import assert from 'node:assert/strict';
import fs from 'node:fs';
import { generate, sample } from './phase-tile.js';
import { samplers } from './procedural.js';

const scores = JSON.parse( fs.readFileSync( new URL( './results.json', import.meta.url ) ) );
const gpu = JSON.parse( fs.readFileSync( new URL( './gpu-results.json', import.meta.url ) ) );
assert.equal( Object.keys( scores.methods ).length, 26 );
for ( const [ name, row ] of Object.entries( scores.methods ) ) {

	assert.equal( Object.keys( row.metrics ).length, 140, name );
	assert.equal( Object.keys( row.image_metrics ).length, 20, name );
	assert.ok( [ ...Object.values( row.metrics ), ...Object.values( row.image_metrics ) ].every( x => Number.isFinite( x ) && x >= 0 ) );

}

assert.deepEqual( scores.methods[ 'FAST-scalar-seed1' ], scores.methods[ 'FAST-scalar-seed2' ] );
assert.equal( gpu.gpu.isFallbackAdapter, false );
for ( const row of gpu.results ) {

	assert.ok( row.elapsedSeconds >= 5 && row.completedFrames > 0 && row.maxCPUError <= .002 );

}

for ( const size of [ 16, 32, 64, 128 ] ) {

	const tile = generate( { size, seed: 1, passes: 2 } );
	assert.deepEqual( Array.from( tile ).sort( ( a, b ) => a - b ), Array.from( { length: size * size }, ( _, i ) => ( i + .5 ) / ( size * size ) ) );
	assert.notEqual( sample( tile, 3, 7, 0 ), sample( tile, 3, 7, 32 ) );

}

for ( const name of [ 'hashed-digital', 'hierarchical-digital' ] ) {

	const values = Array.from( { length: 32 }, ( _, frame ) => Math.floor( samplers[ name ]( 13, 91, frame, 1 ) * 32 ) ).sort( ( a, b ) => a - b );
	assert.deepEqual( values, Array.from( { length: 32 }, ( _, i ) => i ) );

}

for ( const [ name, fn ] of Object.entries( samplers ) ) for ( let i = 0; i < 1000; i ++ ) {

	const value = fn( i % 129, i % 197, i, 2 );
	assert.ok( Number.isFinite( value ) && value >= 0 && value < 1, name );

}

console.log( '26 streams, 160 diagnostics each; hardware throughput, temporal strata and phase histograms verified.' );

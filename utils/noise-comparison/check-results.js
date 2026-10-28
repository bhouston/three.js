import assert from 'node:assert/strict';
import fs from 'node:fs';
import { animatedIGN, tiledBayerWeyl } from './procedural-reference.js';

const data = JSON.parse( fs.readFileSync( new URL( './results.json', import.meta.url ) ) );
assert.deepEqual( data.frames, [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 ] );
for ( const group of [ 'raw', 'blur' ] ) {

	assert.equal( Object.keys( data[ group ] ).length, 6 );
	for ( const values of Object.values( data[ group ] ) ) {

		assert.equal( values.length, 11 );
		assert.ok( values.every( value => Number.isFinite( value ) && value >= 0 ) );

	}

}

assert.equal( data.referenceSamples.length, 24 );
for ( const point of data.referenceSamples ) {

	assert.equal( tiledBayerWeyl( point.x, point.y, point.frame ), point.hierarchy );
	assert.equal( animatedIGN( point.x, point.y, point.frame ), point.ign );

}

console.log( 'Six series / 11 frame counts and all 24 independent NumPy/JS fixtures agree.' );

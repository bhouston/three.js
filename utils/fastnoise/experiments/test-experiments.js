import assert from 'node:assert/strict';
import fs from 'node:fs';
import { generate as multifeature } from './multifeature.js';
import { generate as threshold } from './threshold-refine.js';
import { generate as stratified } from './stratified.js';
import { specialize } from './gpu-features.js';

const options = { size: 4, depth: 4, seed: 2, passes: 2, iterations: 2 };
for ( const generate of [ multifeature, threshold, stratified ] ) {

	const a = generate( options ), b = generate( options );
	assert.equal( a.length, 64 );
	assert.deepEqual( a, b );
	assert.ok( a.every( v => Number.isFinite( v ) && v >= 0 && v < 1 ) );
	assert.throws( () => generate( { size: 0 } ), RangeError );

}

const initial = multifeature( options ), copy = initial.slice();
for ( const generate of [ multifeature, threshold ] ) {

	const result = generate( { ...options, initial } );
	assert.notEqual( result.buffer, initial.buffer );
	assert.deepEqual( initial, copy );

}

// Independently verify cached energy deltas for tiny wrapped kernels.
for ( const size of [ 1, 2, 3 ] ) for ( const preserveColumns of [ true, false ] ) threshold( { size, depth: 3, passes: 4, thresholdCount: 3, preserveColumns, verify: true } );
// Restricting swaps to time preserves the exact original multiset per pixel.
const before = stratified( { ...options, passes: 0 } ), after = stratified( options );
for ( let p = 0; p < 16; p ++ ) {

	const a = [], b = [];
	for ( let z = 0; z < 4; z ++ ) {

		a.push( before[ z * 16 + p ] ); b.push( after[ z * 16 + p ] );

	}

	assert.deepEqual( a.sort(), b.sort() );

}

const source = fs.readFileSync( new URL( '../generator.html', import.meta.url ), 'utf8' );
assert.equal( specialize( source ), source );
assert.throws( () => specialize( source, { identityWeight: - 1 } ), RangeError );
assert.throws( () => specialize( source, { swapFloor: 0 } ), RangeError );
assert.throws( () => specialize( 'changed', {} ) );
const candidate = specialize( source, { identityWeight: .25, frequencyWeight: .05, mixedWeight: .2, swapFloor: 2, stratified: true } );
assert.ok( candidate.includes( 'Experimental per-pixel Latin hypercube' ) );
assert.ok( candidate.includes( 'max( params.suppression, 2u )' ) );
console.log( 'Experiment invariants, cache deltas, histogram preservation, source guards and input immutability passed.' );

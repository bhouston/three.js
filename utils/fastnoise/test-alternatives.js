// CPU invariants; quality comparisons live in compare.py, not these tests.
import assert from 'node:assert/strict';
import { generate as hierarchy } from './alternatives/hierarchy.js';
import { generate as lattice } from './alternatives/lattice.js';
import { generate as relaxation } from './alternatives/relaxation.js';

for ( const [ name, generate ] of Object.entries( { hierarchy, lattice, relaxation } ) ) {

	const options = { size: 16, depth: 32, seed: 37 };
	const a = generate( options ), b = generate( options ), c = generate( { ...options, seed: 38 } );
	assert.ok( a instanceof Float32Array, `${name}: Float32Array output` );
	assert.equal( a.length, 16 * 16 * 32, `${name}: dimensions` );
	assert.deepEqual( a, b, `${name}: reproducibility` );
	assert.ok( a.some( ( value, i ) => value !== c[ i ] ), `${name}: seed affects output` );
	assert.ok( a.every( value => Number.isFinite( value ) && value >= 0 && value < 1 ), `${name}: finite half-open range` );
	const histogram = new Uint32Array( 16 );
	let mean = 0;
	for ( const value of a ) { mean += value / a.length; histogram[ Math.floor( value * 16 ) ] ++; }
	assert.ok( mean > 0.48 && mean < 0.52, `${name}: marginal mean ${mean}` );
	assert.ok( histogram.every( count => count > a.length / 32 ), `${name}: no collapsed histogram strata` );
	for ( const options of [ { size: 0 }, { depth: 0 }, { size: 1.5 }, { depth: NaN } ] ) assert.throws( () => generate( options ), RangeError, `${name}: rejects invalid dimensions` );
	const small = generate( { size: 2, depth: 2, seed: 11 } );
	assert.equal( small.length, 8 );
	assert.ok( small.every( value => Number.isFinite( value ) && value >= 0 && value < 1 ), `${name}: small periodic dimensions` );
	console.log( `${name}: reproducibility, seeds, dimensions, range, histogram and input validation passed` );

}

assert.throws( () => relaxation( { size: 1 } ), RangeError );
assert.throws( () => relaxation( { depth: 1 } ), RangeError );
assert.throws( () => relaxation( { iterations: - 1 } ), RangeError );
assert.throws( () => hierarchy( { size: 4097 } ), RangeError );

// A dyadic temporal block must contain exactly one value in every corresponding
// stratum for each pixel, including blocks beginning after the first frame.
const size = 8, area = size * size, depth = 32;
const volume = hierarchy( { size, depth, seed: 19 } );
for ( const length of [ 2, 4, 8, 16, 32 ] ) {

	for ( let start = 0; start < depth; start += length ) for ( let p = 0; p < area; p ++ ) {

		const strata = new Set();
		for ( let z = start; z < start + length; z ++ ) strata.add( Math.floor( volume[ z * area + p ] * length ) );
		assert.equal( strata.size, length, `hierarchy temporal block ${length}, start ${start}, pixel ${p}` );

	}

}

for ( let z = 0; z < depth; z ++ ) for ( let y = 0; y < size; y += 2 ) for ( let x = 0; x < size; x += 2 ) {

	const strata = new Set();
	for ( let dy = 0; dy < 2; dy ++ ) for ( let dx = 0; dx < 2; dx ++ ) strata.add( Math.floor( volume[ z * area + ( y + dy ) * size + x + dx ] * 4 ) );
	assert.equal( strata.size, 4, `hierarchy spatial quartet at ${x},${y},${z}` );

}

console.log( 'hierarchy: aligned dyadic temporal blocks and spatial quartet strata passed' );

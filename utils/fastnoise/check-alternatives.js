import assert from 'node:assert/strict';
import { generate as local } from './methods/local-energy.js';
import { generate as hierarchy } from './methods/hierarchy.js';
import { generate as filter } from './methods/filter-rank.js';

for ( const generate of [ local, hierarchy, filter ] ) {

	const options = { size: 8, depth: 8, seed: 7 };
	const a = generate( options ), b = generate( options );
	assert.deepEqual( a, b );
	assert.equal( a.length, 512 );
	assert.ok( a.every( x => Number.isFinite( x ) && x >= 0 && x < 1 ) );
	assert.notDeepEqual( a, generate( { ...options, seed: 8 } ) );
	assert.throws( () => generate( { size: 0 } ), RangeError );

}

const h = hierarchy( { size: 8, depth: 8 } );
for ( let p = 0; p < 64; p ++ ) assert.equal( new Set( Array.from( { length: 8 }, ( _, z ) => Math.floor( h[ z * 64 + p ] * 8 ) ) ).size, 8 );
const before = local( { size: 8, depth: 8, passes: 0 } );
const after = local( { size: 8, depth: 8 } );
assert.deepEqual( before.slice().sort(), after.slice().sort() );
// Independent full convolution checks that cached swap decisions lower the
// specified binary-mask objective, including wrapped overlapping stencils.
function objective( volume ) {

	const S = 8, Z = 8, P = 64, radius = 4, alpha = 0.25;
	const temporal = Array.from( { length: radius + 1 }, ( _, i ) => ( 1 - alpha ) ** i );
	const norm = temporal.reduce( ( sum, v ) => sum + v * v, 0 );
	const spatial = [ 1 / 6, 4 / 6, 1, 4 / 6, 1 / 6 ];
	let energy = 0;
	for ( let c = 0; c < 15; c ++ ) {

		const v = volume.map( x => Number( x < ( c + 0.5 ) / 15 ) );
		for ( let z = 0; z < Z; z ++ ) for ( let y = 0; y < S; y ++ ) for ( let x = 0; x < S; x ++ ) {

			const center = v[ z * P + y * S + x ];
			for ( let dy = - 2; dy <= 2; dy ++ ) for ( let dx = - 2; dx <= 2; dx ++ ) energy += center * v[ z * P + ( y + dy + S ) % S * S + ( x + dx + S ) % S ] * spatial[ dy + 2 ] * spatial[ dx + 2 ];
			for ( let dz = - radius; dz <= radius; dz ++ ) {

				let weight = 0;
				for ( let j = 0; j + Math.abs( dz ) <= radius; j ++ ) weight += temporal[ j ] * temporal[ j + Math.abs( dz ) ] / norm;
				energy += center * v[ ( z + dz + Z ) % Z * P + y * S + x ] * weight;

			}

		}

	}

	return energy;

}

assert.ok( objective( after ) < objective( before ) );
console.log( 'Alternative determinism, range, stratification, histogram and objective checks passed.' );

import assert from 'node:assert/strict';
import { generate as multiscale } from './methods/multiscale-energy.js';
import { generate as stratified } from './methods/stratified-refine.js';

const options = { size: 4, depth: 4, thresholdCount: 7, passes: 3, seed: 4 };
const initial = multiscale( { ...options, passes: 0 } );
const refined = multiscale( { ...options, initialVolume: initial } );
assert.deepEqual( initial.slice().sort(), refined.slice().sort() );
assert.deepEqual( refined, multiscale( { ...options, initialVolume: initial } ) );
assert.ok( refined.every( x => x >= 0 && x < 1 && Number.isFinite( x ) ) );
assert.throws( () => multiscale( { size: 0 } ), RangeError );
assert.throws( () => multiscale( { ...options, initialVolume: new Float32Array( 2 ) } ), RangeError );
const before = stratified( { ...options, passes: 0 } );
const after = stratified( options );
assert.deepEqual( after, stratified( options ) );
for ( let p = 0; p < 16; p ++ ) {

	const a = Array.from( { length: 4 }, ( _, z ) => before[ z * 16 + p ] ).sort();
	const b = Array.from( { length: 4 }, ( _, z ) => after[ z * 16 + p ] ).sort();
	assert.deepEqual( a, b ); assert.equal( new Set( b.map( x => Math.floor( x * 4 ) ) ).size, 4 );

}
// Independent convolution-based CDF energy, not the optimizer's cached stencil.
function energy( volume ) {

	const S = 4, Z = 4, P = 16, wrap = v => ( v % 4 + 4 ) % 4;
	let result = 0;
	for ( let c = 0; c < 7; c ++ ) {

		const v = volume.map( x => Number( x < ( c + 0.5 ) / 7 ) );
		for ( const sigma of [ 0.7, 1 ] ) {

			const radius = Math.ceil( 3 * sigma );
			const filter = Array.from( { length: 2 * radius + 1 }, ( _, i ) => Math.exp( - ( ( i - radius ) ** 2 ) / ( 2 * sigma ** 2 ) ) );
			const norm = filter.reduce( ( sum, x ) => sum + x * x, 0 ) ** 2;
			for ( let z = 0; z < Z; z ++ ) for ( let y = 0; y < S; y ++ ) for ( let x = 0; x < S; x ++ ) {

				let value = 0;
				for ( let dy = - radius; dy <= radius; dy ++ ) for ( let dx = - radius; dx <= radius; dx ++ ) value += v[ z * P + wrap( y + dy ) * S + wrap( x + dx ) ] * filter[ dy + radius ] * filter[ dx + radius ];
				result += value * value / norm / 2;

			}

		}
		const temporal = [ 1, 0.9, 0.9 ** 2, 0.9 ** 3 ], norm = temporal.reduce( ( sum, x ) => sum + x * x, 0 );
		for ( let z = 0; z < Z; z ++ ) for ( let p = 0; p < P; p ++ ) {

			let value = 0;
			for ( let k = 0; k < Z; k ++ ) value += v[ wrap( z + k ) * P + p ] * temporal[ k ];
			result += value * value / norm;

		}

	}
	return result;

}
assert.ok( energy( refined ) < energy( initial ) );
console.log( 'Frontier determinism, bounds, histogram/strata and independent CDF energy checks passed.' );

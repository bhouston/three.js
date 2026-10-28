import assert from 'node:assert/strict';
import { generate } from './methods/pareto-energy.js';

const size = 8, depth = 8, plane = size * size;
const options = { size, depth, passes: 4, thresholdCount: 7 };
const initial = generate( { ...options, passes: 0 } );
const saved = initial.slice();
let report;
const output = generate( { ...options, initialVolume: initial, onProgress: value => {

	report = value;

} } );
const wrap = ( value, period ) => ( value % period + period ) % period;
const transforms = [ x => x, x => x * x, x => x ** 4, x => Math.cos( 2 * Math.PI * x ) ];

// Direct convolution, independent of the optimizer's autocorrelation cache.
function energy( volume ) {

	const result = [ Array( 5 ).fill( 0 ), Array( 5 ).fill( 0 ) ];
	const quantized = Float64Array.from( volume, x => ( Math.floor( x * 256 ) + 0.5 ) / 256 );
	const spatial = Array.from( { length: 7 }, ( _, i ) => Math.exp( - ( ( i - 3 ) ** 2 ) / 2 ) );
	const spatialNorm = spatial.reduce( ( sum, x ) => sum + x * x, 0 ) ** 2;
	const temporal = Array.from( { length: depth }, ( _, i ) => 0.9 ** i );
	const temporalNorm = temporal.reduce( ( sum, x ) => sum + x * x, 0 );
	for ( let channel = 0; channel < 11; channel ++ ) {

		const index = channel < 7 ? 0 : channel - 6;
		const values = Float64Array.from( quantized, x => channel < 7 ? Number( x < ( channel + 0.5 ) / 7 ) : transforms[ channel - 7 ]( x ) );
		for ( let z = 0; z < depth; z ++ ) for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

			let spatialValue = 0, temporalValue = 0;
			for ( let dy = - 3; dy <= 3; dy ++ ) for ( let dx = - 3; dx <= 3; dx ++ ) spatialValue += spatial[ dy + 3 ] * spatial[ dx + 3 ] * values[ z * plane + wrap( y + dy, size ) * size + wrap( x + dx, size ) ];
			for ( let d = 0; d < depth; d ++ ) temporalValue += temporal[ d ] * values[ wrap( z + d, depth ) * plane + y * size + x ];
			result[ 0 ][ index ] += spatialValue * spatialValue / spatialNorm;
			result[ 1 ][ index ] += temporalValue * temporalValue / temporalNorm;

		}

	}

	return result;

}

const before = energy( initial ), after = energy( output );
assert.ok( report.accepted > 0, 'Exercise accepted swaps.' );
assert.deepEqual( report.objectives, [ 'cdf', 'identity', 'squared', 'fourth', 'cosine' ] );
for ( let family = 0; family < 2; family ++ ) for ( let channel = 0; channel < 5; channel ++ ) {

	const difference = after[ family ][ channel ] - before[ family ][ channel ];
	assert.ok( difference <= 1e-8, `Objective increased: family ${ family }, channel ${ channel }.` );
	assert.ok( report.objectiveChanges[ family ][ channel ] <= 1e-8 );
	assert.ok( Math.abs( difference - report.objectiveChanges[ family ][ channel ] ) < 1e-8, 'Cached predicted change must match direct convolution.' );

}

assert.deepEqual( initial, saved, 'initialVolume must remain unchanged.' );
assert.deepEqual( initial.slice().sort(), output.slice().sort(), 'Raw histogram must remain exact.' );
assert.deepEqual( output, generate( { ...options, initialVolume: initial } ), 'Refinement must be deterministic.' );
console.log( 'Pareto quantized convolution: all ten objective changes, histogram, determinism and input immutability passed.' );

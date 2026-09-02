import {
	float, vec2, vec3, vec4,
	exp, hash, inversesqrt, tanh
} from 'three/tsl';
import { gpuTest, gpuFuzzTest } from '../tsl/gpu-test-utils.js';
import { geluNew, layerNorm, linear, softmax } from '../../../../examples/jsm/gpgpu/llm/LLMMath.js';

// GPU-native coverage of the GPT-2 math specs (gelu_new, softmax, layer
// norm, Conv1D linear). Expected values come from the CPU helpers in
// LLMMath.js, published closed forms, or independently checkable properties
// (softmax sums to 1, shift invariance, layer-norm mean 0). They are never
// taken by re-running the TSL expression under test -- see
// https://ben3d.ca/blog/the-rise-of-test-theater

const SWIZZLE = [ 'x', 'y', 'z', 'w' ];

function tslVec( values ) {

	if ( values.length === 1 ) return float( values[ 0 ] );
	if ( values.length === 2 ) return vec2( values[ 0 ], values[ 1 ] );
	if ( values.length === 3 ) return vec3( values[ 0 ], values[ 1 ], values[ 2 ] );

	return vec4( values[ 0 ], values[ 1 ], values[ 2 ], values[ 3 ] );

}

function tslSum( value, count ) {

	let sum = value.x;

	for ( let i = 1; i < count; i ++ ) sum = sum.add( value[ SWIZZLE[ i ] ] );

	return sum;

}

function tslMaxComponent( value, count ) {

	let maxValue = value.x;

	for ( let i = 1; i < count; i ++ ) maxValue = maxValue.max( value[ SWIZZLE[ i ] ] );

	return maxValue;

}

// GPT-2 gelu_new (Hendrycks GELU tanh approximation).
function tslGeluNew( x ) {

	const cubic = x.mul( x ).mul( x ).mul( 0.044715 ).add( x );
	const inner = cubic.mul( Math.sqrt( 2 / Math.PI ) );

	return x.mul( 0.5 ).mul( tanh( inner ).add( float( 1 ) ) );

}

// Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x))).
function tslSoftmax( values, count ) {

	const shifted = exp( values.sub( tslMaxComponent( values, count ) ) );

	return shifted.div( tslSum( shifted, count ) );

}

// Layer norm: (x - mean) / sqrt(var + eps) * weight + bias, over the vector.
function tslLayerNorm( input, weight, bias, count, epsilon = 1e-5 ) {

	const mean = tslSum( input, count ).div( count );
	const delta = input.sub( mean );
	const variance = tslSum( delta.mul( delta ), count ).div( count );

	return delta.mul( inversesqrt( variance.add( epsilon ) ) ).mul( weight ).add( bias );

}

// Dense layer with GPT-2 Conv1D / Hugging Face weight layout:
// index = input * outputSize + output.
function tslLinear( input, inputSize, weightRows, bias ) {

	let sum = bias;

	for ( let i = 0; i < inputSize; i ++ ) {

		sum = sum.add( weightRows[ i ].mul( input[ SWIZZLE[ i ] ] ) );

	}

	return sum;

}

function weightRows( weight, inputSize, outputSize ) {

	const rows = [];

	for ( let i = 0; i < inputSize; i ++ ) {

		const row = [];

		for ( let o = 0; o < outputSize; o ++ ) row.push( weight[ i * outputSize + o ] );

		rows.push( tslVec( row ) );

	}

	return rows;

}

function cpuLinear( input, weight, bias ) {

	return linear(
		new Float32Array( input ),
		new Float32Array( weight ),
		bias === null ? null : new Float32Array( bias ),
		input.length,
		bias === null ? weight.length / input.length : bias.length
	);

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'LLMMath (GPU)', () => {

			gpuTest( 'gelu_new matches LLMMath at known values', ( { assert } ) => {

				const cases = [ 0, 0.5, - 0.5, 1, - 1, 2, - 2, 3, - 3, 8, - 8 ];

				for ( const x of cases ) {

					assert.closeAbs( tslGeluNew( float( x ) ), float( geluNew( x ) ), 1e-5, `gelu_new(${ x })` );

				}

				// Published GPT-2 gelu_new(1) reference (independent of both
				// LLMMath.js and the TSL formula above).
				assert.closeAbs( tslGeluNew( float( 1 ) ), float( 0.84119199 ), 1e-5, 'gelu_new(1) matches the published reference' );

			} );

			gpuTest( 'gelu_new matches LLMMath on a dense grid', ( { assert } ) => {

				for ( let i = 0; i <= 32; i ++ ) {

					const x = - 4 + i * 0.25;

					assert.closeAbs( tslGeluNew( float( x ) ), float( geluNew( x ) ), 1e-4, `gelu_new(${ x })` );

				}

			}, { maxAssertions: 40 } );

			gpuTest( 'gelu_new saturates at large |x|', ( { assert } ) => {

				// tanh saturates, so gelu_new(x) -> x for large +x and -> 0
				// for large -x. Checked against LLMMath, not against x itself,
				// so a GPU tanh that failed to saturate would still be caught.
				assert.closeAbs( tslGeluNew( float( 20 ) ), float( geluNew( 20 ) ), 1e-4, 'gelu_new(20)' );
				assert.closeAbs( tslGeluNew( float( - 20 ) ), float( geluNew( - 20 ) ), 1e-4, 'gelu_new(-20)' );
				assert.closeAbs( tslGeluNew( float( 20 ) ), float( 20 ), 1e-3, 'gelu_new(20) ~= 20' );
				assert.closeAbs( tslGeluNew( float( - 20 ) ), float( 0 ), 1e-3, 'gelu_new(-20) ~= 0' );

			} );

			gpuTest( 'softmax matches LLMMath for vec2/vec3/vec4', ( { assert } ) => {

				// exp(k) / (e + e^2 + e^3) for k=1,2,3 -- a published 3-class
				// softmax, independent of both LLMMath.softmax and tslSoftmax.
				assert.closeAbs(
					tslSoftmax( vec3( 1, 2, 3 ), 3 ),
					vec3( 0.09003057, 0.24472848, 0.66524094 ),
					1e-5,
					'softmax([1, 2, 3]) closed form'
				);

				const cases = [
					[ 1, 1, 1 ],
					[ 0, 0, 10 ],
					[ - 1, - 2, - 3 ],
					[ 2, - 1 ],
					[ 0.1, 0.2, 0.3, 0.4 ]
				];

				for ( const values of cases ) {

					const cpu = softmax( new Float32Array( values ) );

					assert.closeAbs(
						tslSoftmax( tslVec( values ), values.length ),
						tslVec( cpu ),
						1e-5,
						`softmax([${ values.join( ', ' ) }])`
					);

				}

			} );

			gpuTest( 'softmax is numerically stable for large logits', ( { assert } ) => {

				// Without the max subtraction, exp(100) overflows. The CPU
				// helper and TSLAttention both subtract max first, so these
				// must match the finite CPU result rather than Inf/NaN.
				const large = [ 100, 101, 102 ];
				const cpu = softmax( new Float32Array( large ) );

				assert.closeAbs( tslSoftmax( vec3( 100, 101, 102 ), 3 ), tslVec( cpu ), 1e-5, 'softmax([100, 101, 102])' );
				assert.closeAbs(
					tslSoftmax( vec3( 100, 101, 102 ), 3 ),
					tslSoftmax( vec3( 0, 1, 2 ), 3 ),
					1e-5,
					'softmax([100, 101, 102]) == softmax([0, 1, 2]) -- shift invariance at large magnitude'
				);

			} );

			gpuTest( 'layerNorm matches LLMMath for vec2/vec3/vec4', ( { assert } ) => {

				// mean=2, var=2/3, (x-2)/sqrt(2/3+1e-5) -- independent of both
				// layerNorm() and tslLayerNorm().
				assert.closeAbs(
					tslLayerNorm( vec3( 1, 2, 3 ), vec3( 1, 1, 1 ), vec3( 0, 0, 0 ), 3 ),
					vec3( - 1.2247356, 0, 1.2247356 ),
					1e-4,
					'layerNorm([1, 2, 3]) closed form'
				);

				const cases = [
					{ input: [ 1, 2, 3 ], weight: [ 2, 0.5, 1 ], bias: [ 0.1, - 0.2, 0.3 ] },
					{ input: [ - 2, 4 ], weight: [ 1, 1 ], bias: [ 0, 0 ] },
					{ input: [ 0.5, - 1.5, 2, 0 ], weight: [ 1, 1, 1, 1 ], bias: [ 0, 0, 0, 0 ] },
					{ input: [ 5, 5, 5 ], weight: [ 1, 1, 1 ], bias: [ 0, 0, 0 ] }
				];

				for ( const { input, weight, bias } of cases ) {

					const cpu = layerNorm(
						new Float32Array( input ),
						new Float32Array( weight ),
						new Float32Array( bias )
					);

					assert.closeAbs(
						tslLayerNorm( tslVec( input ), tslVec( weight ), tslVec( bias ), input.length ),
						tslVec( cpu ),
						1e-4,
						`layerNorm([${ input.join( ', ' ) }])`
					);

				}

			} );

			gpuTest( 'layerNorm respects a non-default epsilon', ( { assert } ) => {

				const input = [ 1, 1, 1.0001 ];
				const weight = [ 1, 1, 1 ];
				const bias = [ 0, 0, 0 ];
				const epsilon = 1e-2;
				const cpu = layerNorm(
					new Float32Array( input ),
					new Float32Array( weight ),
					new Float32Array( bias ),
					epsilon
				);

				assert.closeAbs(
					tslLayerNorm( tslVec( input ), tslVec( weight ), tslVec( bias ), 3, epsilon ),
					tslVec( cpu ),
					1e-4,
					'layerNorm with epsilon=1e-2'
				);

			} );

			gpuTest( 'linear matches LLMMath (GPT-2 Conv1D weight layout)', ( { assert } ) => {

				// 1*3 + 2*5 + 7 == 20, 1*4 + 2*6 + 8 == 24 -- hand-computed,
				// not taken from linear() or from the TSL expression.
				assert.closeAbs(
					tslLinear( vec2( 1, 2 ), 2, [ vec2( 3, 4 ), vec2( 5, 6 ) ], vec2( 7, 8 ) ),
					vec2( 20, 24 ),
					1e-4,
					'2x2 with bias, hand-computed'
				);

				const cases = [
					{ input: [ 1, 2 ], weight: [ 3, 4, 5, 6 ], bias: [ 0, 0 ], label: '2x2 zero bias' },
					{ input: [ 1, 2 ], weight: [ 1, 0, 0, 1 ], bias: [ 0, 0 ], label: '2x2 identity' },
					{ input: [ 0, 0 ], weight: [ 3, 4, 5, 6 ], bias: [ 7, 8 ], label: 'zero input returns bias' },
					{ input: [ 1, 2, 3 ], weight: [ 1, 0, 0, 1, 0, 0 ], bias: [ 0, 0 ], label: '3->2' },
					{ input: [ 1, - 1 ], weight: [ 1, 2, 3, 4, 5, 6 ], bias: [ 0.5, - 0.5, 1 ], label: '2->3' }
				];

				for ( const { input, weight, bias, label } of cases ) {

					const cpu = cpuLinear( input, weight, bias );

					assert.closeAbs(
						tslLinear( tslVec( input ), input.length, weightRows( weight, input.length, bias.length ), tslVec( bias ) ),
						tslVec( cpu ),
						1e-4,
						label
					);

				}

			} );

			gpuTest( 'linear without bias matches LLMMath', ( { assert } ) => {

				// 1*3 + 2*5 == 13, 1*4 + 2*6 == 16.
				assert.closeAbs(
					tslLinear( vec2( 1, 2 ), 2, weightRows( [ 3, 4, 5, 6 ], 2, 2 ), vec2( 0, 0 ) ),
					vec2( 13, 16 ),
					1e-4,
					'null bias is treated as zeros'
				);

			} );

			gpuFuzzTest( 'softmax outputs are a probability distribution', 256, ( { instanceIndex, assert } ) => {

				const logits = vec3(
					hash( instanceIndex.add( 1 ) ).mul( 4 ).sub( 2 ),
					hash( instanceIndex.add( 1000 ) ).mul( 4 ).sub( 2 ),
					hash( instanceIndex.add( 2000 ) ).mul( 4 ).sub( 2 )
				);
				const probabilities = tslSoftmax( logits, 3 );

				assert.closeAbs( tslSum( probabilities, 3 ), float( 1 ), 1e-4, 'softmax sums to 1' );
				assert.greaterThan( probabilities, vec3( 0, 0, 0 ), 'every softmax component is positive' );

			} );

			gpuFuzzTest( 'softmax is invariant to a uniform logit shift', 256, ( { instanceIndex, assert } ) => {

				const logits = vec3(
					hash( instanceIndex.add( 1 ) ).mul( 6 ).sub( 3 ),
					hash( instanceIndex.add( 1000 ) ).mul( 6 ).sub( 3 ),
					hash( instanceIndex.add( 2000 ) ).mul( 6 ).sub( 3 )
				);

				assert.closeAbs(
					tslSoftmax( logits, 3 ),
					tslSoftmax( logits.add( 10 ), 3 ),
					1e-4,
					'softmax(v) == softmax(v + 10)'
				);

			} );

			gpuFuzzTest( 'layerNorm with unit affine has mean 0', 256, ( { instanceIndex, assert } ) => {

				const input = vec3(
					hash( instanceIndex.add( 1 ) ).mul( 4 ).sub( 2 ),
					hash( instanceIndex.add( 1000 ) ).mul( 4 ).sub( 2 ),
					hash( instanceIndex.add( 2000 ) ).mul( 4 ).sub( 2 )
				);
				const normalized = tslLayerNorm( input, vec3( 1, 1, 1 ), vec3( 0, 0, 0 ), 3 );

				assert.closeAbs( tslSum( normalized, 3 ).div( 3 ), float( 0 ), 1e-4, 'mean(layerNorm(x)) ~= 0' );

			} );

			gpuFuzzTest( 'linear without bias is homogeneous: f(2x) == 2 f(x)', 256, ( { instanceIndex, assert } ) => {

				const input = vec2(
					hash( instanceIndex.add( 1 ) ).mul( 4 ).sub( 2 ),
					hash( instanceIndex.add( 1000 ) ).mul( 4 ).sub( 2 )
				);
				const rows = [ vec2( 0.5, - 1.5 ), vec2( 2, 0.25 ) ];
				const doubled = tslLinear( input.mul( 2 ), 2, rows, vec2( 0, 0 ) );
				const scaled = tslLinear( input, 2, rows, vec2( 0, 0 ) ).mul( 2 );

				assert.closeAbs( doubled, scaled, 1e-4, 'f(2x) == 2 f(x) when bias is 0' );

			} );

		} );

	} );

} );

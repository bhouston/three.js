import { describe, expect, it } from 'vitest';
import { activate, createMLP, forwardMLP, powerLog, sigmoid } from '../../../../examples/jsm/ntc/training/NTCMLP.js';

describe( 'Addons > Neural > NeuralMLP', () => {

	describe( 'activate', () => {

		it( 'relu passes positive values through unchanged', () => {

			expect( activate( 2.5, 'relu' ) ).toBe( 2.5 );

		} );

		it( 'relu clamps negative values to zero', () => {

			expect( activate( - 1.5, 'relu' ) ).toBe( 0 );

		} );

		it( 'relu maps exactly zero to zero', () => {

			expect( activate( 0, 'relu' ) ).toBe( 0 );

		} );

		it( 'leakyRelu passes positive values through unchanged', () => {

			expect( activate( 4, 'leakyRelu' ) ).toBe( 4 );

		} );

		it( 'leakyRelu scales negative values by 0.01 instead of zeroing them', () => {

			expect( activate( - 4, 'leakyRelu' ) ).toBeCloseTo( - 0.04, 10 );

		} );

		it( 'leakyRelu treats zero as the non-negative branch', () => {

			expect( activate( 0, 'leakyRelu' ) ).toBe( 0 );

		} );

		it( 'tanh saturates toward +/-1 for large magnitudes', () => {

			expect( activate( 10, 'tanh' ) ).toBeCloseTo( 1, 5 );
			expect( activate( - 10, 'tanh' ) ).toBeCloseTo( - 1, 5 );

		} );

		it( 'tanh is zero at zero', () => {

			expect( activate( 0, 'tanh' ) ).toBe( 0 );

		} );

		it( 'linear passes any value through unchanged', () => {

			expect( activate( 2, 'linear' ) ).toBe( 2 );
			expect( activate( - 37.5, 'linear' ) ).toBe( - 37.5 );

		} );

		it( 'falls back to linear (unchanged) for an unrecognized activation name', () => {

			expect( activate( 8, 'not-a-real-activation' ) ).toBe( 8 );

		} );

	} );

	describe( 'sigmoid', () => {

		it( 'is exactly 0.5 at zero', () => {

			expect( sigmoid( 0 ) ).toBeCloseTo( 0.5, 10 );

		} );

		it( 'approaches 1 for large positive input', () => {

			expect( sigmoid( 50 ) ).toBeCloseTo( 1, 8 );

		} );

		it( 'approaches 0 for large negative input without producing NaN', () => {

			expect( sigmoid( - 50 ) ).toBeCloseTo( 0, 8 );
			expect( Number.isFinite( sigmoid( - 1000 ) ) ).toBe( true );

		} );

		it( 'stays within (0, 1) across a spread of inputs', () => {

			for ( const value of [ - 20, - 1, - 0.001, 0.001, 1, 20 ] ) {

				const result = sigmoid( value );
				expect( result ).toBeGreaterThan( 0 );
				expect( result ).toBeLessThan( 1 );

			}

		} );

	} );

	describe( 'powerLog', () => {

		it( 'is zero at value = 1 regardless of power', () => {

			expect( powerLog( 1, 3 ) ).toBeCloseTo( 0, 10 );
			expect( powerLog( 1, 0.5 ) ).toBeCloseTo( 0, 10 );

		} );

		it( 'reduces to (value - 1) when power = 1', () => {

			expect( powerLog( 5, 1 ) ).toBeCloseTo( 4, 10 );

		} );

	} );

	describe( 'createMLP', () => {

		it( 'builds one layer per adjacent size pair, including hidden layers', () => {

			const mlp = createMLP( 4, [ 8, 6 ], 2, () => 0.5 );

			expect( mlp.layers.length ).toBe( 3 ); // 4->8, 8->6, 6->2
			expect( mlp.layers[ 0 ].inputSize ).toBe( 4 );
			expect( mlp.layers[ 0 ].outputSize ).toBe( 8 );
			expect( mlp.layers[ 1 ].inputSize ).toBe( 8 );
			expect( mlp.layers[ 1 ].outputSize ).toBe( 6 );
			expect( mlp.layers[ 2 ].inputSize ).toBe( 6 );
			expect( mlp.layers[ 2 ].outputSize ).toBe( 2 );

		} );

		it( 'supports zero hidden layers (a single linear/whatever layer)', () => {

			const mlp = createMLP( 3, [], 1, () => 0.5 );

			expect( mlp.layers.length ).toBe( 1 );
			expect( mlp.layers[ 0 ].inputSize ).toBe( 3 );
			expect( mlp.layers[ 0 ].outputSize ).toBe( 1 );

		} );

		it( 'sizes every layer\'s weights/biases arrays to input*output / output', () => {

			const mlp = createMLP( 5, [ 7 ], 2, () => 0.5 );

			expect( mlp.layers[ 0 ].weights.length ).toBe( 5 * 7 );
			expect( mlp.layers[ 0 ].biases.length ).toBe( 7 );
			expect( mlp.layers[ 1 ].weights.length ).toBe( 7 * 2 );
			expect( mlp.layers[ 1 ].biases.length ).toBe( 2 );

		} );

		it( 'assigns the requested hidden/output activation names to the right layers', () => {

			const mlp = createMLP( 2, [ 3 ], 1, () => 0.5, 'tanh', 'sigmoid' );

			expect( mlp.layers[ 0 ].activation ).toBe( 'tanh' );
			expect( mlp.layers[ 1 ].activation ).toBe( 'sigmoid' );

		} );

		it( 'defaults to relu hidden / linear output activations when unspecified', () => {

			const mlp = createMLP( 2, [ 3 ], 1, () => 0.5 );

			expect( mlp.layers[ 0 ].activation ).toBe( 'relu' );
			expect( mlp.layers[ 1 ].activation ).toBe( 'linear' );

		} );

		it( 'is deterministic given the same seeded random function', () => {

			let calls = 0;
			const random = () => {

				calls += 1;
				// deterministic pseudo-sequence, not just a constant, to also
				// exercise weight variety
				return ( Math.sin( calls ) + 1 ) / 2;

			};

			const mlpA = createMLP( 4, [ 5 ], 2, random );
			calls = 0;
			const mlpB = createMLP( 4, [ 5 ], 2, random );

			expect( mlpA.layers[ 0 ].weights ).toEqual( mlpB.layers[ 0 ].weights );
			expect( mlpA.layers[ 1 ].weights ).toEqual( mlpB.layers[ 1 ].weights );

		} );

		it( 'gives a 3-wide linear output layer a shared positive gray bias (RGB head)', () => {

			const mlp = createMLP( 32, [ 32 ], 3, () => 0.75, 'relu', 'linear' );
			const outputLayer = mlp.layers[ 1 ];

			expect( outputLayer.biases ).toEqual( [ 0.3, 0.3, 0.3 ] );

		} );

		it( 'does not apply the RGB gray-bias treatment to a non-3-wide linear output', () => {

			const mlp = createMLP( 32, [ 32 ], 4, () => 0.75, 'relu', 'linear' );
			const outputLayer = mlp.layers[ 1 ];

			expect( outputLayer.biases ).toEqual( [ 0, 0, 0, 0 ] );

		} );

		it( 'does not apply the RGB gray-bias treatment when the output activation is not linear', () => {

			const mlp = createMLP( 32, [ 32 ], 3, () => 0.75, 'relu', 'sigmoid' );
			const outputLayer = mlp.layers[ 1 ];

			expect( outputLayer.biases ).toEqual( [ 0, 0, 0 ] );

		} );

		it( 'scales hidden-layer weight magnitude by He initialization (sqrt(2/fanIn))', () => {

			const random = () => 1; // pins every weight to its maximum magnitude: +scale
			const mlp = createMLP( 16, [ 8 ], 1, random, 'relu', 'linear' );
			const heScale = Math.sqrt( 2 / 16 );

			for ( const weight of mlp.layers[ 0 ].weights ) {

				expect( weight ).toBeCloseTo( heScale, 10 );

			}

		} );

	} );

	describe( 'forwardMLP', () => {

		it( 'tracks one activation snapshot per layer boundary (input + each layer output)', () => {

			const mlp = createMLP( 2, [ 3 ], 1, () => 0.75, 'relu', 'linear' );
			const run = forwardMLP( mlp, [ 1, 0.5 ] );

			expect( run.activations.length ).toBe( 3 ); // input, hidden, output
			expect( run.preActivations.length ).toBe( 2 ); // one per real layer
			expect( run.output.length ).toBe( 1 );

		} );

		it( 'does not mutate the input array it was given', () => {

			const mlp = createMLP( 2, [ 2 ], 1, () => 0.5 );
			const input = [ 1, 2 ];
			forwardMLP( mlp, input );

			expect( input ).toEqual( [ 1, 2 ] );

		} );

		it( 'computes an all-zero-weight network as pure bias output', () => {

			const mlp = createMLP( 3, [ 4 ], 2, () => 0 ); // random() = 0 => every weight is 0
			mlp.layers[ 0 ].biases = [ 1, 1, 1, 1 ];
			mlp.layers[ 1 ].biases = [ 2, - 2 ];

			const run = forwardMLP( mlp, [ 5, - 5, 5 ] );

			// hidden = relu(bias) = relu(1) = 1 for every hidden unit regardless of input
			// output = bias only, since hidden-to-output weights are also 0
			expect( run.output ).toEqual( [ 2, - 2 ] );

		} );

		it( 'matches a hand-computed single-layer linear result', () => {

			const random = () => 0.75; // createMLP maps this to a fixed weight value
			const mlp = createMLP( 2, [], 1, random, 'relu', 'linear' );
			const weight = mlp.layers[ 0 ].weights;
			const bias = mlp.layers[ 0 ].biases[ 0 ];

			const input = [ 2, - 1 ];
			const expected = bias + weight[ 0 ] * input[ 0 ] + weight[ 1 ] * input[ 1 ];

			const run = forwardMLP( mlp, input );

			expect( run.output[ 0 ] ).toBeCloseTo( expected, 10 );

		} );

		it( 'applies relu so a negative pre-activation is clamped in the recorded activation', () => {

			const mlp = createMLP( 1, [ 1 ], 1, () => 0, 'relu', 'linear' );
			mlp.layers[ 0 ].weights = [ - 1 ];
			mlp.layers[ 0 ].biases = [ 0 ];
			mlp.layers[ 1 ].weights = [ 1 ];
			mlp.layers[ 1 ].biases = [ 0 ];

			const run = forwardMLP( mlp, [ 5 ] ); // pre-activation = -5, relu'd to 0

			expect( run.preActivations[ 0 ] ).toEqual( [ - 5 ] );
			expect( run.activations[ 1 ] ).toEqual( [ 0 ] );
			expect( run.output ).toEqual( [ 0 ] );

		} );

		it( 'handles a zero-length input (no input features) without throwing', () => {

			const mlp = createMLP( 0, [ 2 ], 1, () => 0.5 );
			const run = forwardMLP( mlp, [] );

			expect( run.output.length ).toBe( 1 );
			expect( run.output.every( Number.isFinite ) ).toBe( true );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import {
	createMLP,
	forwardMLP,
	activate,
	sigmoid,
	powerLog
} from '../../../../examples/jsm/neural/NeuralMLP.js';

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralMLP', () => {

			test( 'evaluates activation functions', () => {

				expect( activate( 2.5, 'relu' ) ).toBe( 2.5 );
				expect( activate( - 1.5, 'relu' ) ).toBe( 0 );

				expect( activate( 2.0, 'linear' ) ).toBe( 2.0 );

				expect( Math.abs( sigmoid( 0 ) - 0.5 ) < 1e-6 ).toBeTruthy();
				expect( powerLog( 1, 3 ) ).toBe( 0 );

			} );

			test( 'performs forward propagation', () => {

				const random = () => 0.75;
				const mlp = createMLP( 2, [ 3 ], 1, random, 'relu', 'linear' );

				const input = [ 1, 0.5 ];
				const run = forwardMLP( mlp, input );

				expect( run.activations.length ).toBe( 3 );
				expect( run.output.length ).toBe( 1 );

			} );

			test( 'initializes a linear RGB head with a shared gray bias', () => {

				const random = () => 0.75;
				const mlp = createMLP( 32, [ 32 ], 3, random, 'relu', 'linear' );
				const hiddenLayer = mlp.layers[ 0 ];
				const outputLayer = mlp.layers[ 1 ];
				const heScale = Math.sqrt( 2 / 32 );
				const rgbScale = heScale * 0.45;

				expect( outputLayer.biases ).toEqual( [ 0.3, 0.3, 0.3 ] );
				expect( hiddenLayer.weights.every( ( weight ) => Math.abs( weight ) <= heScale + 1e-12 ) ).toBeTruthy();
				expect( outputLayer.weights.every( ( weight ) => Math.abs( weight ) <= rgbScale + 1e-12 ) ).toBeTruthy();
				expect( outputLayer.weights.some( ( weight ) => Math.abs( weight ) > rgbScale * 0.4 ) ).toBeTruthy();

			} );

		} );

	} );

} );

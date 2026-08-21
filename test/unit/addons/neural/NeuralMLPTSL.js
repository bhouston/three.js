import { describe, test, expect } from 'vitest';
import { Matrix4 } from 'three';
import {
	packLayerWeightsMat4,
	packLayerBiasesVec4
} from '../../../../examples/jsm/neural/NeuralMLPTSL.js';

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralMLPTSL', () => {

			test( 'packs layer weights into mat4 uniform array', () => {

				// 6 inputs (ceil(6/4) = 2 input vectors), 5 outputs (ceil(5/4) = 2
				// output vectors) => a 2x2 grid of mat4 blocks.
				const inputSize = 6;
				const outputSize = 5;
				const weights = new Array( outputSize * inputSize );

				for ( let o = 0; o < outputSize; o ++ ) {

					for ( let i = 0; i < inputSize; i ++ ) {

						weights[ o * inputSize + i ] = o * inputSize + i + 1;

					}

				}

				const packed = packLayerWeightsMat4( weights, inputSize, outputSize );

				expect( packed.length ).toBe( 4 );
				expect( packed[ 0 ] instanceof Matrix4 ).toBeTruthy();

				// Block (outputVector=0, inputVector=0): outputs 0..3, inputs 0..3 -
				// fully in range, no padding.
				const block00 = new Matrix4().set(
					1, 2, 3, 4,
					7, 8, 9, 10,
					13, 14, 15, 16,
					19, 20, 21, 22
				);
				expect( packed[ 0 ].equals( block00 ) ).toBeTruthy();

				// Block (outputVector=0, inputVector=1): outputs 0..3, inputs 4..5 -
				// columns 2/3 zero-padded (inputs 6/7 don't exist).
				const block01 = new Matrix4().set(
					5, 6, 0, 0,
					11, 12, 0, 0,
					17, 18, 0, 0,
					23, 24, 0, 0
				);
				expect( packed[ 1 ].equals( block01 ) ).toBeTruthy();

				// Block (outputVector=1, inputVector=0): output 4 only - rows 1..3
				// zero-padded (outputs 5/6/7 don't exist).
				const block10 = new Matrix4().set(
					25, 26, 27, 28,
					0, 0, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0
				);
				expect( packed[ 2 ].equals( block10 ) ).toBeTruthy();

				// Block (outputVector=1, inputVector=1): fully padded except the
				// single (output 4, input 4/5) entries.
				const block11 = new Matrix4().set(
					29, 30, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0
				);
				expect( packed[ 3 ].equals( block11 ) ).toBeTruthy();

			} );

			test( 'packs layer biases into vec4 uniform array', () => {

				const biases = [ 1, 2, 3, 4, 5 ];
				const packed = packLayerBiasesVec4( biases );

				expect( packed.length ).toBe( 2 );
				expect( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ] ).toEqual( [ 1, 2, 3, 4 ] );
				expect( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ] ).toEqual( [ 5, 0, 0, 0 ] );

			} );

		} );

	} );

} );

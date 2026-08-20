import { describe, test, expect } from 'vitest';
import { Vector4 } from 'three';
import {
	packLayerWeightsVec4,
	packLayerBiasesVec4
} from '../../../../examples/jsm/neural/NeuralMLPTSL.js';

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralMLPTSL', () => {

			test( 'packs layer weights into vec4 uniform array', () => {

				// 6 inputs (ceil(6/4) = 2 vec4s per output) and 2 outputs => 4 Vector4s
				const weights = [
					1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12
				];
				const packed = packLayerWeightsVec4( weights, 6, 2 );

				expect( packed.length ).toBe( 4 );
				expect( packed[ 0 ] instanceof Vector4 ).toBeTruthy();
				expect( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ] ).toEqual( [ 1, 2, 3, 4 ] );
				expect( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ] ).toEqual( [ 5, 6, 0, 0 ] );
				expect( [ packed[ 2 ].x, packed[ 2 ].y, packed[ 2 ].z, packed[ 2 ].w ] ).toEqual( [ 7, 8, 9, 10 ] );
				expect( [ packed[ 3 ].x, packed[ 3 ].y, packed[ 3 ].z, packed[ 3 ].w ] ).toEqual( [ 11, 12, 0, 0 ] );

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

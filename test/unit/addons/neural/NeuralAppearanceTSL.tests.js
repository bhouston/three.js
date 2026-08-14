import { Vector4 } from 'three';
import {
	packLayerWeights,
	packLayerBiases
} from '../../../../examples/jsm/neural/NeuralAppearanceTSL.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceTSL', () => {

			QUnit.test( 'packs layer weights into vec4 uniform array', ( assert ) => {

				// 6 inputs (ceil(6/4) = 2 vec4s per output) and 2 outputs => 4 Vector4s
				const weights = [
					1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12
				];
				const packed = packLayerWeights( weights, 6, 2 );

				assert.strictEqual( packed.length, 4, 'packs into 4 Vector4s' );
				assert.ok( packed[ 0 ] instanceof Vector4, 'elements are Vector4 instances' );
				assert.deepEqual( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ], [ 1, 2, 3, 4 ], 'first vector contains first 4 weights' );
				assert.deepEqual( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ], [ 5, 6, 0, 0 ], 'second vector zero-padded' );
				assert.deepEqual( [ packed[ 2 ].x, packed[ 2 ].y, packed[ 2 ].z, packed[ 2 ].w ], [ 7, 8, 9, 10 ], 'third vector starts output 1' );
				assert.deepEqual( [ packed[ 3 ].x, packed[ 3 ].y, packed[ 3 ].z, packed[ 3 ].w ], [ 11, 12, 0, 0 ], 'fourth vector zero-padded' );

			} );

			QUnit.test( 'packs layer biases into vec4 uniform array', ( assert ) => {

				const biases = [ 1, 2, 3, 4, 5 ];
				const packed = packLayerBiases( biases );

				assert.strictEqual( packed.length, 2, 'packs 5 biases into 2 Vector4s' );
				assert.deepEqual( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ], [ 1, 2, 3, 4 ], 'first vector filled' );
				assert.deepEqual( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ], [ 5, 0, 0, 0 ], 'second vector zero-padded' );

			} );

		} );

	} );

} );

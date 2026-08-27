import { describe, expect, it } from 'vitest';
import {
	computeDecoderInputBound,
	computeLayerPreActivationBound,
	computeActivationOutputBound,
	packLayerWeightsInt8,
	buildInt8HiddenLayers
} from '../../../../examples/jsm/ntc/NTCMLPQuantization.js';
import { quantizeSymmetricInt8 } from '../../../../examples/jsm/ntc/NTCPackedDotProduct.js';
import { hardGELU } from '../../../../examples/jsm/ntc/training/NTCMLP.js';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';

describe( 'Addons > NTC > NTCMLPQuantization', () => {

	describe( 'computeDecoderInputBound', () => {

		it( 'is at least 1 (covers LOD/positional-encoding\'s [0,1]/[-1,1] range) even for an all-zero grid', () => {

			const model = createNTCGridPyramidModel( { channels: 2, levels: 1, baseResolution: 4, hiddenSizes: [ 3 ], outputChannels: 2 }, () => 0.5 ); // random()=0.5 -> all-zero grids

			expect( computeDecoderInputBound( model ) ).toBe( 1 );

		} );

		it( 'reflects the actual maximum absolute latent value when it exceeds 1', () => {

			const model = createNTCGridPyramidModel( { channels: 1, levels: 1, baseResolution: 2, hiddenSizes: [ 2 ], outputChannels: 1 }, () => 0.5 );
			model.grids[ 0 ].data.set( [ 0.2, - 3.7, 0.1, 1.5 ] );

			// grid.data is a Float32Array, so 3.7 round-trips as
			// 3.700000047683716 - a low tolerance here checks the right value
			// was found, not exact float32/float64 bit equality.
			expect( computeDecoderInputBound( model ) ).toBeCloseTo( 3.7, 5 );

		} );

	} );

	describe( 'computeLayerPreActivationBound', () => {

		it( 'matches a hand-computed bound for a tiny 2-input, 2-output layer', () => {

			const layer = {
				inputSize: 2, outputSize: 2,
				weights: [ 1, - 2, 0.5, 0.5 ], // output0: [1,-2], output1: [0.5,0.5]
				biases: [ 0.1, - 0.2 ]
			};
			const inputBound = 3;

			// output0: |0.1| + 3*(|1|+|-2|) = 0.1 + 9 = 9.1
			// output1: |-0.2| + 3*(|0.5|+|0.5|) = 0.2 + 3 = 3.2
			expect( computeLayerPreActivationBound( layer, inputBound ) ).toBeCloseTo( 9.1, 10 );

		} );

		it( 'is 0 for an all-zero layer', () => {

			const layer = { inputSize: 2, outputSize: 1, weights: [ 0, 0 ], biases: [ 0 ] };
			expect( computeLayerPreActivationBound( layer, 5 ) ).toBe( 0 );

		} );

		it( 'is a genuinely sound (never-exceeded) bound across many random inputs within +/-inputBound', () => {

			const layer = {
				inputSize: 3, outputSize: 4,
				weights: [ 0.3, - 1.2, 0.7, - 0.4, 0.9, 0.1, 1.5, - 0.6, 0.2, 0.05, - 0.05, 2.0 ],
				biases: [ 0.1, - 0.3, 0, 0.5 ]
			};
			const inputBound = 2.5;
			const bound = computeLayerPreActivationBound( layer, inputBound );

			// Random sampling of real inputs within [-inputBound, inputBound]^3 -
			// none should ever exceed the computed bound.
			let seed = 42;
			const random = () => {

				seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
				return seed / 0x7fffffff;

			};

			for ( let trial = 0; trial < 200; trial ++ ) {

				const x = [ 0, 1, 2 ].map( () => ( random() * 2 - 1 ) * inputBound );

				for ( let j = 0; j < layer.outputSize; j ++ ) {

					let z = layer.biases[ j ];
					for ( let i = 0; i < layer.inputSize; i ++ ) z += layer.weights[ j * layer.inputSize + i ] * x[ i ];
					expect( Math.abs( z ) ).toBeLessThanOrEqual( bound + 1e-9 );

				}

			}

		} );

	} );

	describe( 'computeActivationOutputBound', () => {

		it( 'relu: returns the pre-activation bound unchanged', () => {

			expect( computeActivationOutputBound( 4.2, 'relu' ) ).toBe( 4.2 );

		} );

		it( 'hgelu: is at least 1.5 even for a small pre-activation bound', () => {

			expect( computeActivationOutputBound( 0.1, 'hgelu' ) ).toBe( 1.5 );

		} );

		it( 'hgelu: matches the pre-activation bound when it already exceeds 1.5', () => {

			expect( computeActivationOutputBound( 10, 'hgelu' ) ).toBe( 10 );

		} );

		it( 'hgelu: is a genuinely sound bound - hardGELU never exceeds it in absolute value for |x| <= preActivationBound', () => {

			for ( const preActivationBound of [ 0.2, 0.75, 1.5, 3, 10 ] ) {

				const bound = computeActivationOutputBound( preActivationBound, 'hgelu' );

				for ( let i = 0; i <= 200; i ++ ) {

					const x = - preActivationBound + ( 2 * preActivationBound * i / 200 );
					expect( Math.abs( hardGELU( x ) ) ).toBeLessThanOrEqual( bound + 1e-9 );

				}

			}

		} );

	} );

	describe( 'packLayerWeightsInt8', () => {

		it( 'produces outputSize * ceil(inputSize/4) packed u32 words', () => {

			const layer = { inputSize: 5, outputSize: 3, weights: new Array( 15 ).fill( 0.1 ), biases: [ 0, 0, 0 ] };
			const packed = packLayerWeightsInt8( layer, 0.01 );

			expect( packed.length ).toBe( 3 * Math.ceil( 5 / 4 ) );

		} );

		it( 'quantizes each weight the same way quantizeSymmetricInt8 would, packed 4 per group', () => {

			const layer = {
				inputSize: 4, outputSize: 1,
				weights: [ 1, - 1, 0.5, - 0.5 ],
				biases: [ 0 ]
			};
			const scale = 0.01;
			const packed = packLayerWeightsInt8( layer, scale );

			const expectedLevels = layer.weights.map( ( w ) => quantizeSymmetricInt8( w, scale ) );
			// Unpack the single group manually (little-endian signed bytes).
			const word = packed[ 0 ];
			const unpacked = [ 0, 1, 2, 3 ].map( ( i ) => {

				const byte = ( word >>> ( 8 * i ) ) & 0xFF;
				return byte >= 128 ? byte - 256 : byte;

			} );

			expect( unpacked ).toEqual( expectedLevels );

		} );

	} );

	describe( 'buildInt8HiddenLayers', () => {

		it( 'produces one entry per hidden layer, excluding the final (output) layer', () => {

			const model = createNTCGridPyramidModel( { channels: 2, levels: 1, baseResolution: 4, hiddenSizes: [ 3, 5 ], outputChannels: 2 }, () => 0.75 );
			const int8Layers = buildInt8HiddenLayers( model );

			expect( int8Layers.length ).toBe( 2 ); // hiddenSizes.length, not decoder.layers.length (3)
			expect( int8Layers[ 0 ].outputSize ).toBe( 3 );
			expect( int8Layers[ 1 ].outputSize ).toBe( 5 );

		} );

		it( 'each entry carries a positive inputScale/weightScale and correctly-sized packedWeights/biases', () => {

			const model = createNTCGridPyramidModel( { channels: 4, levels: 1, baseResolution: 4, hiddenSizes: [ 6 ], outputChannels: 3 }, () => 0.9 );
			const [ layer ] = buildInt8HiddenLayers( model );

			expect( layer.inputScale ).toBeGreaterThan( 0 );
			expect( layer.weightScale ).toBeGreaterThan( 0 );
			expect( layer.packedWeights.length ).toBe( layer.outputSize * layer.groupCount );
			expect( layer.biases.length ).toBe( layer.outputSize );
			expect( layer.groupCount ).toBe( Math.ceil( layer.inputSize / 4 ) );

		} );

		it( 'a later layer\'s inputScale is derived from the previous layer\'s propagated output bound, not always equal to the first layer\'s', () => {

			const model = createNTCGridPyramidModel( { channels: 4, levels: 1, baseResolution: 4, hiddenSizes: [ 4, 4 ], outputChannels: 3, hiddenActivation: 'relu' }, () => 0.95 ); // large-magnitude weights
			const int8Layers = buildInt8HiddenLayers( model );

			// With non-trivial weights, the propagated bound after layer 0 should
			// generally differ from the raw decoder input bound used for layer 0
			// itself - if this test's model happens to produce an unchanged
			// scale, the seeded random() above should be adjusted, but for a
			// well-conditioned layer with real weights this reliably differs.
			expect( int8Layers[ 1 ].inputScale ).not.toBeCloseTo( int8Layers[ 0 ].inputScale, 10 );

		} );

	} );

} );

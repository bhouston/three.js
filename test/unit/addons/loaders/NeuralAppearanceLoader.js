import { describe, test, expect } from 'vitest';
import { DataTexture, DataUtils } from 'three';
import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';
import { encodeUint8Base64, encodeFloat16Base64, encodeMLPLayersBase64 } from '../../../../examples/jsm/neural/NeuralBinaryCodec.js';

// Builds a compact (uint8-quantized latents, float16-packed MLP weights)
// `.neuralAppearance` manifest by hand, mirroring exactly what
// NeuralAppearanceManifest.js's `compactAppearanceJson` produces from a real
// trained model - see that file for the authoritative shape.

function encodeLevel( data ) {

	const min = Math.min( ...data );
	const max = Math.max( ...data );

	return {
		width: 1,
		height: 1,
		channels: 4,
		wrap: 'repeat',
		dtype: 'uint8',
		min,
		max,
		dataBase64: encodeUint8Base64( Float32Array.from( data ), min, max )
	};

}

function encodeHead( inputSize, layers, outputActivation, extra = {} ) {

	return {
		inputSize,
		mlp: encodeMLPLayersBase64( layers ),
		outputActivation,
		...extra
	};

}

function encodeRotation( inputSize, outputSize, weights ) {

	return {
		inputSize,
		outputSize,
		dtype: 'float16',
		dataBase64: encodeFloat16Base64( Float32Array.from( weights ) )
	};

}

function createBrdfWeights() {

	const weights = new Array( 84 ).fill( 0 );
	weights[ 0 ] = 1; // output 0 <- latent channel 0 (level 0, channel 0)
	weights[ 29 ] = 1; // output 1 <- latent channel 1 (level 0, channel 1)
	weights[ 83 ] = 1; // output 2 <- frame1's wo.n projection (== 1 for wo = [0,0,1] with zero rotation weights)
	return weights;

}

function createManifest() {

	return {
		format: 'three-neural-appearance',
		version: 9,
		name: 'unit test material',
		latents: {
			channelsPerLevel: 4,
			wrap: 'repeat',
			levels: [
				encodeLevel( [ 0.8, 0.2, 0.1, 0.5 ] ),
				encodeLevel( [ 0.3, 0.4, 0.5, 0.6 ] ),
				encodeLevel( [ 0, 0, 0, 0 ] ),
				encodeLevel( [ 0, 0, 0, 0 ] )
			]
		},
		outputs: {
			brdf: {
				inputSize: 28,
				rotation: encodeRotation( 16, 12, new Array( 192 ).fill( 0 ) ),
				...encodeHead( 28, [
					{ inputSize: 28, outputSize: 3, activation: 'linear', biases: [ 0, 0, 0 ], weights: createBrdfWeights() }
				], { type: 'linear' } )
			},
			ibl: encodeHead( 22, [
				{ inputSize: 22, outputSize: 4, activation: 'linear', biases: [ 0, 0, 1, 0 ], weights: new Array( 22 * 4 ).fill( 0 ) }
			], { type: 'linear' } ),
			indirectRadiance: encodeHead( 22, [
				{ inputSize: 22, outputSize: 3, activation: 'linear', biases: [ 0, 0, 0 ], weights: new Array( 22 * 3 ).fill( 0 ) }
			], { type: 'linear' } ),
			indirectIrradiance: encodeHead( 22, [
				{ inputSize: 22, outputSize: 3, activation: 'linear', biases: [ 0, 0, 0 ], weights: new Array( 22 * 3 ).fill( 0 ) }
			], { type: 'linear' } ),
			emission: encodeHead( 16, [
				{
					inputSize: 16, outputSize: 3, activation: 'linear', biases: [ 0, 0, 0 ], weights: [
						1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
					]
				}
			], { type: 'linear' } ),
			opacity: encodeHead( 16, [
				{ inputSize: 16, outputSize: 1, activation: 'linear', biases: [ 0 ], weights: [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] }
			], { type: 'sigmoid' }, { alphaCutoff: 0.5 } )
		},
		referenceEvaluations: [
			{
				uv: [ 0.5, 0.5 ],
				wi: [ 0, 0, 1 ],
				wo: [ 0, 0, 1 ],
				rgb: [ 0.8, 0.2, 1 ]
			}
		]
	};

}

function cpuEval( data, reference ) {

	const inputs = [];

	for ( const texture of data.latentTextures ) {

		const image = texture.image.data;
		inputs.push(
			DataUtils.fromHalfFloat( image[ 0 ] ), DataUtils.fromHalfFloat( image[ 1 ] ),
			DataUtils.fromHalfFloat( image[ 2 ] ), DataUtils.fromHalfFloat( image[ 3 ] )
		);

	}

	inputs.push(
		reference.wi[ 0 ], reference.wi[ 1 ], reference.wi[ 2 ],
		reference.wo[ 0 ], reference.wo[ 1 ], reference.wo[ 2 ],
		reference.wi[ 0 ], reference.wi[ 1 ], reference.wi[ 2 ],
		reference.wo[ 0 ], reference.wo[ 1 ], reference.wo[ 2 ]
	);

	return evaluateDecoderLayers( data.outputs.brdf.layers, inputs );

}

function evaluateDecoderLayers( layers, inputs ) {

	let values = inputs;

	for ( const layer of layers ) {

		const next = [];

		for ( let output = 0; output < layer.outputSize; output ++ ) {

			let value = layer.biases[ output ];

			for ( let input = 0; input < layer.inputSize; input ++ ) {

				value += layer.weights[ output * layer.inputSize + input ] * values[ input ];

			}

			next.push( layer.activation === 'relu' ? Math.max( 0, value ) : value );

		}

		values = next;

	}

	return values.map( ( value ) => Math.max( 0, value ) );

}

function closeToArray( actual, expected, tolerance = 1e-2 ) {

	for ( let i = 0; i < actual.length; i ++ ) {

		expect( Math.abs( actual[ i ] - expected[ i ] ) < tolerance ).toBeTruthy();

	}

}

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'NeuralAppearanceLoader', () => {

			test( 'parses neural appearance material data', () => {

				const loader = new NeuralAppearanceLoader();
				const data = loader.parse( createManifest() );

				expect( data.isNeuralAppearanceData ).toBe( true );
				expect( data.latentTextures.length ).toBe( 4 );
				expect( data.latentTextures[ 0 ] instanceof DataTexture ).toBeTruthy();
				expect( data.outputs.brdf.layers.length ).toBe( 1 );
				expect( data.outputs.ibl.layers.length ).toBe( 1 );
				expect( data.outputs.emission.layers.length ).toBe( 1 );
				expect( data.outputs.opacity.mode ).toBe( 'mask' );
				expect( data.outputs.opacity.alphaCutoff ).toBe( 0.5 );

			} );

			test( 'parses explicit blend and mask opacity modes', () => {

				const loader = new NeuralAppearanceLoader();
				const blendManifest = createManifest();
				blendManifest.outputs.opacity.mode = 'blend';
				const blendData = loader.parse( blendManifest );

				expect( blendData.outputs.opacity.mode ).toBe( 'blend' );

				const maskManifest = createManifest();
				maskManifest.outputs.opacity.mode = 'mask';
				const maskData = loader.parse( maskManifest );

				expect( maskData.outputs.opacity.mode ).toBe( 'mask' );

			} );

			test( 'rejects unsupported opacity mode', () => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.outputs.opacity.mode = 'invalid_mode';

				expect( () => loader.parse( manifest ) ).toThrow( /Unsupported outputs\.opacity\.mode/ );

			} );

			test( 'rejects unsupported format versions', () => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.version = 1;

				expect( () => loader.parse( manifest ) ).toThrow( /Unsupported version/ );

			} );

			test( 'rejects a decoder mlp block whose declared layout disagrees with the head\'s own inputSize', () => {

				// Compact `mlp.layout` entries (`{ rows, cols, kind }`, see
				// NeuralBinaryCodec.js's encodeMLPLayersBase64/
				// decodeMLPLayersBase64) are self-consistent by construction - a
				// decoded layer's weights.length is always exactly
				// layout.rows * layout.cols, so it can no longer disagree with
				// itself the way a hand-edited plain `weights` array used to.
				// What can still disagree is the *declared* head.inputSize
				// against the first decoded layer's own inputSize (rows) - see
				// normalizeOutputHead's cross-check in NeuralAppearanceLoader.js.
				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.outputs.brdf.mlp.layout[ 0 ].rows = 27;

				expect( () => loader.parse( manifest ) ).toThrow( /outputs\.brdf\.layers\[0\]\.inputSize must be/ );

			} );

			test( 'rejects a latent grid level count that does not match the output heads\' declared input sizes', () => {

				// The number of latent grid levels itself is no longer fixed (see
				// computeGridLevels/NeuralAppearanceTrainer - models can be trained
				// with any level count), so popping a level is only invalid because
				// this manifest's output heads still declare inputSize for the
				// *original* 4-level shape (28/22/16) rather than the 3-level shape
				// (24/18/12) that would actually match - see normalizeOutputs.
				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.latents.levels.pop();

				expect( () => loader.parse( manifest ) ).toThrow( /outputs\.brdf\.inputSize must be/ );

			} );

			test( 'matches exported reference evaluation within uint8/float16 quantization tolerance', () => {

				const loader = new NeuralAppearanceLoader();
				const data = loader.parse( createManifest() );
				const reference = data.referenceEvaluations[ 0 ];

				closeToArray( cpuEval( data, reference ), reference.rgb );

			} );

		} );

	} );

} );

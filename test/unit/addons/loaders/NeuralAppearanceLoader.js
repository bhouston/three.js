import { describe, test, expect } from 'vitest';
import { DataTexture, DataUtils } from 'three';
import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';

function createLevel( data ) {

	return { width: 1, height: 1, wrap: 'repeat', data };

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
		version: 7,
		name: 'unit test material',
		latents: {
			channelsPerLevel: 4,
			wrap: 'repeat',
			levels: [
				createLevel( [ 0.8, 0.2, 0.1, 0.5 ] ),
				createLevel( [ 0.3, 0.4, 0.5, 0.6 ] ),
				createLevel( [ 0, 0, 0, 0 ] ),
				createLevel( [ 0, 0, 0, 0 ] )
			]
		},
		outputs: {
			brdf: {
				inputSize: 28,
				rotation: {
					inputSize: 16,
					outputSize: 12,
					weights: new Array( 192 ).fill( 0 )
				},
				layers: [
					{
						inputSize: 28,
						outputSize: 3,
						activation: 'linear',
						biases: [ 0, 0, 0 ],
						weights: createBrdfWeights()
					}
				],
				outputActivation: { type: 'linear' }
			},
			ibl: {
				inputSize: 22,
				layers: [
					{
						inputSize: 22,
						outputSize: 4,
						activation: 'linear',
						biases: [ 0, 0, 1, 0 ],
						weights: new Array( 22 * 4 ).fill( 0 )
					}
				],
				outputActivation: { type: 'linear' }
			},
			indirectRadiance: {
				inputSize: 22,
				layers: [
					{
						inputSize: 22,
						outputSize: 3,
						activation: 'linear',
						biases: [ 0, 0, 0 ],
						weights: new Array( 22 * 3 ).fill( 0 )
					}
				],
				outputActivation: { type: 'linear' }
			},
			indirectIrradiance: {
				inputSize: 22,
				layers: [
					{
						inputSize: 22,
						outputSize: 3,
						activation: 'linear',
						biases: [ 0, 0, 0 ],
						weights: new Array( 22 * 3 ).fill( 0 )
					}
				],
				outputActivation: { type: 'linear' }
			},
			emission: {
				inputSize: 16,
				layers: [
					{
						inputSize: 16,
						outputSize: 3,
						activation: 'linear',
						biases: [ 0, 0, 0 ],
						weights: [
							1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
							0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
							0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
						]
					}
				],
				outputActivation: { type: 'linear' }
			},
			opacity: {
				inputSize: 16,
				layers: [
					{
						inputSize: 16,
						outputSize: 1,
						activation: 'linear',
						biases: [ 0 ],
						weights: [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
					}
				],
				outputActivation: { type: 'sigmoid' },
				alphaCutoff: 0.5
			}
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

function closeToArray( actual, expected, tolerance = 1e-3 ) {

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

			test( 'rejects malformed decoder dimensions', () => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.outputs.brdf.layers[ 0 ].weights.pop();

				expect( () => loader.parse( manifest ) ).toThrow( /weights length/ );

			} );

			test( 'rejects a mismatched number of latent grid levels', () => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.latents.levels.pop();

				expect( () => loader.parse( manifest ) ).toThrow( /latents\.levels must contain/ );

			} );

			test( 'matches exported reference evaluation', () => {

				const loader = new NeuralAppearanceLoader();
				const data = loader.parse( createManifest() );
				const reference = data.referenceEvaluations[ 0 ];

				closeToArray( cpuEval( data, reference ), reference.rgb );

			} );

		} );

	} );

} );

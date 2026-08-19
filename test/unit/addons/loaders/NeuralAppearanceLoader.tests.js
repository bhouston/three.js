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

function closeToArray( assert, actual, expected, message, tolerance = 1e-3 ) {

	for ( let i = 0; i < actual.length; i ++ ) {

		assert.ok( Math.abs( actual[ i ] - expected[ i ] ) < tolerance, `${ message } ${ i }: ${ actual[ i ] } ~= ${ expected[ i ] }` );

	}

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Loaders', () => {

		QUnit.module( 'NeuralAppearanceLoader', () => {

			QUnit.test( 'parses neural appearance material data', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const data = loader.parse( createManifest() );

				assert.strictEqual( data.isNeuralAppearanceData, true, 'marks parsed data' );
				assert.strictEqual( data.latentTextures.length, 4, 'four latent grid level textures' );
				assert.ok( data.latentTextures[ 0 ] instanceof DataTexture, 'creates a data texture' );
				assert.strictEqual( data.outputs.brdf.layers.length, 1, 'brdf layers' );
				assert.strictEqual( data.outputs.ibl.layers.length, 1, 'ibl layers' );
				assert.strictEqual( data.outputs.emission.layers.length, 1, 'emission layers' );
				assert.strictEqual( data.outputs.opacity.mode, 'mask', 'defaults opacity mode to mask' );
				assert.strictEqual( data.outputs.opacity.alphaCutoff, 0.5, 'opacity cutoff' );

			} );

			QUnit.test( 'parses explicit blend and mask opacity modes', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const blendManifest = createManifest();
				blendManifest.outputs.opacity.mode = 'blend';
				const blendData = loader.parse( blendManifest );

				assert.strictEqual( blendData.outputs.opacity.mode, 'blend', 'parses blend opacity mode' );

				const maskManifest = createManifest();
				maskManifest.outputs.opacity.mode = 'mask';
				const maskData = loader.parse( maskManifest );

				assert.strictEqual( maskData.outputs.opacity.mode, 'mask', 'parses mask opacity mode' );

			} );

			QUnit.test( 'rejects unsupported opacity mode', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.outputs.opacity.mode = 'invalid_mode';

				assert.throws( () => loader.parse( manifest ), /Unsupported outputs\.opacity\.mode/, 'throws on invalid opacity mode' );

			} );

			QUnit.test( 'rejects unsupported format versions', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.version = 1;

				assert.throws( () => loader.parse( manifest ), /Unsupported version/, 'throws on unsupported version' );

			} );

			QUnit.test( 'rejects malformed decoder dimensions', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.outputs.brdf.layers[ 0 ].weights.pop();

				assert.throws( () => loader.parse( manifest ), /weights length/, 'throws on invalid weight count' );

			} );

			QUnit.test( 'rejects a mismatched number of latent grid levels', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.latents.levels.pop();

				assert.throws( () => loader.parse( manifest ), /latents\.levels must contain/, 'throws when the level count does not match LATENT_CHANNELS / CHANNELS_PER_LEVEL' );

			} );

			QUnit.test( 'matches exported reference evaluation', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const data = loader.parse( createManifest() );
				const reference = data.referenceEvaluations[ 0 ];

				closeToArray( assert, cpuEval( data, reference ), reference.rgb, 'reference rgb' );

			} );

		} );

	} );

} );

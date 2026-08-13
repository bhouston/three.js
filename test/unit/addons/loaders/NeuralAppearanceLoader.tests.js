import { DataTexture, DataUtils } from 'three';
import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';
import { NeuralAppearanceTrainer, createPhysicalMaterialTeacher, evaluatePhysicalBRDF, generateTrainingSamples } from '../../../../examples/jsm/materials/NeuralAppearanceTrainer.js';

function createManifest() {

	return {
		format: 'three-neural-appearance',
		version: 1,
		name: 'unit test material',
		latents: {
			channels: 8,
			wrap: 'repeat',
			textures: [
				{
					wrap: 'repeat',
					mipmaps: [
						{ width: 1, height: 1, data: [ 0.8, 0.2, 0.1, 0.5 ] }
					]
				},
				{
					wrap: 'repeat',
					mipmaps: [
						{ width: 1, height: 1, data: [ 0.3, 0.4, 0.5, 0.6 ] }
					]
				}
			]
		},
		decoder: {
			inputSize: 20,
			rotation: {
				inputSize: 8,
				outputSize: 12,
				weights: new Array( 96 ).fill( 0 )
			},
			layers: [
				{
					inputSize: 20,
					outputSize: 3,
					activation: 'linear',
					biases: [ 0, 0, 0 ],
					weights: [
						1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
					]
				}
			],
			outputActivation: { type: 'linear' }
		},
		referenceEvaluations: [
			{
				uv: [ 0.5, 0.5 ],
				wi: [ 0, 0, 1 ],
				wo: [ 0, 0, 1 ],
				mip: 0,
				rgb: [ 0.8, 0.2, 1 ]
			}
		]
	};

}

function cpuEval( data, reference ) {

	const texel0 = data.latentTextures[ 0 ].image.data;
	const texel1 = data.latentTextures[ 1 ].image.data;
	const inputs = [
		DataUtils.fromHalfFloat( texel0[ 0 ] ), DataUtils.fromHalfFloat( texel0[ 1 ] ), DataUtils.fromHalfFloat( texel0[ 2 ] ), DataUtils.fromHalfFloat( texel0[ 3 ] ),
		DataUtils.fromHalfFloat( texel1[ 0 ] ), DataUtils.fromHalfFloat( texel1[ 1 ] ), DataUtils.fromHalfFloat( texel1[ 2 ] ), DataUtils.fromHalfFloat( texel1[ 3 ] ),
		reference.wi[ 0 ], reference.wi[ 1 ], reference.wi[ 2 ],
		reference.wo[ 0 ], reference.wo[ 1 ], reference.wo[ 2 ],
		reference.wi[ 0 ], reference.wi[ 1 ], reference.wi[ 2 ],
		reference.wo[ 0 ], reference.wo[ 1 ], reference.wo[ 2 ]
	];
	return evaluateDecoderLayers( data.decoder.layers, inputs );

}

function cpuEvalJson( json, reference ) {

	const latents0 = json.latents.textures[ 0 ].mipmaps[ 0 ].data;
	const latents1 = json.latents.textures[ 1 ].mipmaps[ 0 ].data;
	const inputs = [
		...latents0,
		...latents1,
		reference.wi[ 0 ], reference.wi[ 1 ], reference.wi[ 2 ],
		reference.wo[ 0 ], reference.wo[ 1 ], reference.wo[ 2 ],
		reference.wi[ 0 ], reference.wi[ 1 ], reference.wi[ 2 ],
		reference.wo[ 0 ], reference.wo[ 1 ], reference.wo[ 2 ]
	];

	return evaluateDecoderLayers( json.decoder.layers, inputs );

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

function assertFiniteArray( assert, value, message ) {

	assert.ok( Array.isArray( value ), `${message} is an array` );

	for ( const item of value ) {

		assert.ok( Number.isFinite( item ), `${message} contains finite values` );

	}

}

function createConstantTrainingMaterial() {

	return {
		color: 0x93604a,
		roughness: 0.8,
		metalness: 0,
		specularIntensity: 1
	};

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Loaders', () => {

		QUnit.module( 'NeuralAppearanceLoader', () => {

			QUnit.test( 'parses neural appearance material data', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const data = loader.parse( createManifest() );

				assert.strictEqual( data.isNeuralAppearanceData, true, 'marks parsed data' );
				assert.strictEqual( data.latentTextures.length, 2, 'two latent textures' );
				assert.ok( data.latentTextures[ 0 ] instanceof DataTexture, 'creates a data texture' );
				assert.strictEqual( data.decoder.layers.length, 1, 'decoder layers' );

			} );

			QUnit.test( 'rejects unsupported format versions', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.version = 2;

				assert.throws( () => loader.parse( manifest ), /Unsupported version/, 'throws on unsupported version' );

			} );

			QUnit.test( 'rejects malformed decoder dimensions', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const manifest = createManifest();
				manifest.decoder.layers[ 0 ].weights.pop();

				assert.throws( () => loader.parse( manifest ), /weights length/, 'throws on invalid weight count' );

			} );

			QUnit.test( 'matches exported reference evaluation', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const data = loader.parse( createManifest() );
				const reference = data.referenceEvaluations[ 0 ];

				closeToArray( assert, cpuEval( data, reference ), reference.rgb, 'reference rgb' );

			} );

			QUnit.test( 'samples physical teacher for browser training', ( assert ) => {

				const teacher = createPhysicalMaterialTeacher( {
					color: 0x808080,
					roughness: 0.5,
					metalness: 0
				} );
				const samples = generateTrainingSamples( { batchSize: 16, iterations: 1, colorAugmentation: false }, teacher, () => 0.5 );
				const rgb = evaluatePhysicalBRDF( teacher, [ 0, 0, 1 ], [ 0, 0, 1 ] );

				assert.strictEqual( samples.length, 16, 'generates requested batch size' );
				assert.strictEqual( samples[ 0 ].encoderInputs.length, 14, 'packs encoder inputs' );
				assert.ok( rgb[ 0 ] > 0, 'evaluates non-zero teacher response' );

				for ( const sample of samples ) {

					assert.ok( sample.wi[ 2 ] > 0, 'incident direction is above the surface' );
					assert.ok( sample.wo[ 2 ] >= 0, 'outgoing direction is above the surface' );
					assert.ok( sample.uv[ 0 ] >= 0 && sample.uv[ 0 ] <= 1, 'u coordinate is normalized' );
					assert.ok( sample.uv[ 1 ] >= 0 && sample.uv[ 1 ] <= 1, 'v coordinate is normalized' );
					assert.strictEqual( sample.encoderInputs.length, 14, 'packs 14 encoder inputs' );
					assertFiniteArray( assert, sample.target, 'sample target' );

				}

			} );

			QUnit.test( 'exports trained browser neural appearance data', async ( assert ) => {

				const trainer = new NeuralAppearanceTrainer( {
					resolution: 1,
					iterations: 2,
					batchSize: 4,
					seed: 3,
					yieldEvery: 0
				} );
				const result = await trainer.train( {
					material: {
						color: 0x93604a,
						roughness: 0.45,
						metalness: 0,
						clearcoat: 0.1
					}
				} );
				const data = new NeuralAppearanceLoader().parse( JSON.parse( JSON.stringify( result.json ) ) );

				assert.strictEqual( result.json.format, 'three-neural-appearance', 'exports neural appearance manifest' );
				assert.ok( Array.isArray( result.json.decoder.rotation.weights ), 'keeps rotation weights as JSON arrays' );
				assert.ok( Array.isArray( result.json.latents.textures[ 0 ].mipmaps[ 0 ].data ), 'keeps latent data as JSON arrays' );
				assert.strictEqual( data.latentTextures.length, 2, 'exported manifest parses' );
				assert.strictEqual( data.decoder.layers[ 0 ].inputSize, 20, 'decoder matches runtime input size' );
				assert.strictEqual( result.json.referenceEvaluations.length, 3, 'stores reference evaluations' );

			} );

			QUnit.test( 'reduces loss for a deterministic constant material', async ( assert ) => {

				const losses = [];
				const trainer = new NeuralAppearanceTrainer( {
					resolution: 1,
					iterations: 50,
					batchSize: 16,
					seed: 7,
					yieldEvery: 0,
					colorAugmentation: false
				} );

				await trainer.train( {
					material: createConstantTrainingMaterial(),
					onProgress: ( progress ) => losses.push( progress.loss )
				} );

				assert.ok( losses[ losses.length - 1 ] < losses[ 0 ] * 0.5, `loss decreases from ${losses[ 0 ]} to ${losses[ losses.length - 1 ]}` );

			} );

			QUnit.test( 'trained decoder matches physical BRDF reference points', async ( assert ) => {

				const trainer = new NeuralAppearanceTrainer( {
					resolution: 1,
					iterations: 100,
					batchSize: 64,
					seed: 7,
					yieldEvery: 0,
					colorAugmentation: false
				} );
				const result = await trainer.train( {
					material: createConstantTrainingMaterial()
				} );
				const references = [
					{ wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] },
					{ wi: [ 0.3, 0.1, 0.948683298 ], wo: [ - 0.2, 0.4, 0.89442719 ] },
					{ wi: [ - 0.4, 0.2, 0.89442719 ], wo: [ 0.2, - 0.1, 0.974679434 ] }
				];

				for ( const reference of references ) {

					const actual = cpuEvalJson( result.json, reference );
					const expected = evaluatePhysicalBRDF( result.teacher, reference.wi, reference.wo );

					closeToArray( assert, actual, expected, 'decoder BRDF', 0.07 );

				}

			} );

		} );

	} );

} );

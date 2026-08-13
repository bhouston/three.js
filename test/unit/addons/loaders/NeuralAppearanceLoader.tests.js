import { DataTexture, DataUtils } from 'three';
import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';

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
	const layer = data.decoder.layers[ 0 ];
	const rgb = [];

	for ( let output = 0; output < layer.outputSize; output ++ ) {

		let value = layer.biases[ output ];

		for ( let input = 0; input < layer.inputSize; input ++ ) {

			value += layer.weights[ output * layer.inputSize + input ] * inputs[ input ];

		}

		rgb.push( Math.max( 0, value ) );

	}

	return rgb;

}

function closeToArray( assert, actual, expected, message ) {

	for ( let i = 0; i < actual.length; i ++ ) {

		assert.ok( Math.abs( actual[ i ] - expected[ i ] ) < 1e-3, `${ message } ${ i }: ${ actual[ i ] } ~= ${ expected[ i ] }` );

	}

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

		} );

	} );

} );

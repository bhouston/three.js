import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';
import { NeuralAppearanceNodeMaterial } from '../../../../examples/jsm/neural/NeuralAppearanceNodeMaterial.js';

function createManifest( fill = 0 ) {

	return {
		format: 'three-neural-appearance',
		version: 2,
		name: 'unit test material',
		latents: {
			channels: 8,
			wrap: 'repeat',
			textures: [
				{
					wrap: 'repeat',
					mipmaps: [
						{ width: 1, height: 1, data: [ fill, 0, 0, 0 ] }
					]
				},
				{
					wrap: 'repeat',
					mipmaps: [
						{ width: 1, height: 1, data: [ 0, 0, 0, 0 ] }
					]
				}
			]
		},
		outputs: {
			brdf: {
				inputSize: 20,
				rotation: {
					inputSize: 8,
					outputSize: 12,
					weights: new Array( 96 ).fill( fill )
				},
				layers: [
					{
						inputSize: 20,
						outputSize: 3,
						activation: 'linear',
						biases: [ fill, 0, 0 ],
						weights: new Array( 60 ).fill( fill )
					}
				],
				outputActivation: { type: 'linear' }
			}
		}
	};

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceNodeMaterial', () => {

			QUnit.test( 'updates matching weights without rebuilding', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const textures = material.neuralAppearanceData.latentTextures;
				const weightVector = material._outputUniforms.brdf.layers[ 0 ].weights.array[ 0 ];
				const next = loader.parse( createManifest( 0.25 ) );

				assert.strictEqual( material.updateFromData( next ), true, 'reuses the compiled layout' );
				assert.strictEqual( material.neuralAppearanceData.latentTextures, textures, 'keeps the original latent textures' );
				assert.strictEqual( material._outputUniforms.brdf.layers[ 0 ].weights.array[ 0 ], weightVector, 'keeps the original weight uniforms' );
				assert.notEqual( weightVector.x, 0, 'writes the new packed weights' );

			} );

			QUnit.test( 'rejects a different decoder layout', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const nextManifest = createManifest( 0 );
				nextManifest.outputs.brdf.layers[ 0 ].outputSize = 8;
				nextManifest.outputs.brdf.layers[ 0 ].weights = new Array( 160 ).fill( 0 );
				nextManifest.outputs.brdf.layers[ 0 ].biases = new Array( 8 ).fill( 0 );

				assert.strictEqual( material.updateFromData( loader.parse( nextManifest ) ), false, 'requires a new material' );

			} );

		} );

	} );

} );

import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';
import { NeuralAppearanceNodeMaterial } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceNodeMaterial.js';
import { evaluateNeuralDebugShading } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTSL.js';

function createManifest( fill = 0 ) {

	return {
		format: 'three-neural-appearance',
		version: 6,
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
			},
			ibl: {
				inputSize: 14,
				layers: [
					{
						inputSize: 14,
						outputSize: 4,
						activation: 'linear',
						biases: [ 0, 0, 1, 0 ],
						weights: new Array( 14 * 4 ).fill( fill )
					}
				],
				outputActivation: { type: 'linear' }
			},
			indirectRadiance: {
				inputSize: 14,
				layers: [
					{
						inputSize: 14,
						outputSize: 3,
						activation: 'linear',
						biases: [ 0, 0, 0 ],
						weights: new Array( 14 * 3 ).fill( fill )
					}
				],
				outputActivation: { type: 'linear' }
			},
			indirectIrradiance: {
				inputSize: 14,
				layers: [
					{
						inputSize: 14,
						outputSize: 3,
						activation: 'linear',
						biases: [ 0, 0, 0 ],
						weights: new Array( 14 * 3 ).fill( fill )
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
				const weightVector = material._outputUniforms.brdf.parameters.array[ material._outputUniforms.brdf.layers[ 0 ].weightsOffset ];
				const next = loader.parse( createManifest( 0.25 ) );

				assert.strictEqual( material.updateFromData( next ), true, 'reuses the compiled layout' );
				assert.strictEqual( material.neuralAppearanceData.latentTextures, textures, 'keeps the original latent textures' );
				assert.strictEqual( material._outputUniforms.brdf.parameters.array[ material._outputUniforms.brdf.layers[ 0 ].weightsOffset ], weightVector, 'keeps the original weight uniforms' );
				assert.notEqual( weightVector.x, 0, 'writes the new packed weights' );

			} );

			QUnit.test( 'rejects a different decoder layout', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const nextManifest = createManifest( 0 );
				nextManifest.outputs.brdf.layers = [
					{
						inputSize: 20,
						outputSize: 8,
						activation: 'relu',
						biases: new Array( 8 ).fill( 0 ),
						weights: new Array( 160 ).fill( 0 )
					},
					{
						inputSize: 8,
						outputSize: 3,
						activation: 'linear',
						biases: new Array( 3 ).fill( 0 ),
						weights: new Array( 24 ).fill( 0 )
					}
				];

				assert.strictEqual( material.updateFromData( loader.parse( nextManifest ) ), false, 'requires a new material' );

			} );

			QUnit.test( 'configures mask and blend opacity runtime state', ( assert ) => {

				const loader = new NeuralAppearanceLoader();

				const maskManifest = createManifest( 0 );
				maskManifest.outputs.opacity = {
					inputSize: 8,
					layers: [
						{
							inputSize: 8,
							outputSize: 1,
							activation: 'linear',
							biases: [ 0 ],
							weights: [ 1, 0, 0, 0, 0, 0, 0, 0 ]
						}
					],
					outputActivation: { type: 'sigmoid' },
					mode: 'mask',
					alphaCutoff: 0.5
				};

				const maskMaterial = new NeuralAppearanceNodeMaterial( loader.parse( maskManifest ) );
				assert.strictEqual( maskMaterial.transparent, false, 'mask mode remains opaque' );
				assert.strictEqual( maskMaterial.alphaTest, 0.5, 'mask mode sets alphaTest threshold' );
				assert.ok( maskMaterial.alphaTestNode, 'mask mode assigns alphaTestNode' );
				assert.ok( maskMaterial.opacityNode, 'mask mode assigns opacityNode' );
				assert.ok( maskMaterial.maskNode, 'mask mode assigns maskNode' );

				const blendManifest = createManifest( 0 );
				blendManifest.outputs.opacity = {
					inputSize: 8,
					layers: [
						{
							inputSize: 8,
							outputSize: 1,
							activation: 'linear',
							biases: [ 0 ],
							weights: [ 1, 0, 0, 0, 0, 0, 0, 0 ]
						}
					],
					outputActivation: { type: 'sigmoid' },
					mode: 'blend'
				};

				const blendMaterial = new NeuralAppearanceNodeMaterial( loader.parse( blendManifest ) );
				assert.strictEqual( blendMaterial.transparent, true, 'blend mode sets transparent to true' );
				assert.strictEqual( blendMaterial.depthWrite, false, 'blend mode sets depthWrite to false' );
				assert.ok( blendMaterial.opacityNode, 'blend mode assigns opacityNode' );
				assert.strictEqual( blendMaterial.alphaTestNode, null, 'blend mode does not set alphaTestNode' );
				assert.strictEqual( blendMaterial.maskNode, null, 'blend mode does not set maskNode' );

				assert.strictEqual( maskMaterial.updateFromData( loader.parse( blendManifest ) ), false, 'rebuilds when opacity mode changes from mask to blend' );

			} );

			QUnit.test( 'builds debug shading graphs for decoder-frame visualization', ( assert ) => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const debug = evaluateNeuralDebugShading( material );

				assert.strictEqual( material.debugView, 'shaded', 'defaults to shaded lighting' );
				assert.ok( debug.viewNormal, 'exposes a view-space decoder-frame normal' );
				assert.ok( debug.viewReflect, 'exposes a view-space decoder-frame reflection' );
				assert.ok( debug.viewIrradiance, 'exposes a view-space IBL irradiance query direction' );
				assert.ok( debug.roughness, 'exposes learned IBL specular roughness' );

				material.debugView = 'roughness';
				assert.strictEqual( material.debugView, 'roughness', 'accepts a roughness debug view' );
				material.debugView = 'ibl';
				assert.strictEqual( material.debugView, 'ibl', 'accepts an IBL debug view' );

			} );

		} );

	} );

} );

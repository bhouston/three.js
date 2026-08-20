import { describe, test, expect } from 'vitest';
import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';
import { NeuralAppearanceNodeMaterial } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceNodeMaterial.js';
import { evaluateNeuralDebugShading } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTSL.js';

function createLevel( fill ) {

	return { width: 1, height: 1, wrap: 'repeat', data: [ fill, 0, 0, 0 ] };

}

function createManifest( fill = 0 ) {

	return {
		format: 'three-neural-appearance',
		version: 7,
		name: 'unit test material',
		latents: {
			levels: [ createLevel( fill ), createLevel( 0 ), createLevel( 0 ), createLevel( 0 ) ],
			channelsPerLevel: 4,
			wrap: 'repeat'
		},
		outputs: {
			brdf: {
				inputSize: 28,
				rotation: {
					inputSize: 16,
					outputSize: 12,
					weights: new Array( 192 ).fill( fill )
				},
				layers: [
					{
						inputSize: 28,
						outputSize: 3,
						activation: 'linear',
						biases: [ fill, 0, 0 ],
						weights: new Array( 84 ).fill( fill )
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
						weights: new Array( 22 * 4 ).fill( fill )
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
						weights: new Array( 22 * 3 ).fill( fill )
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
						weights: new Array( 22 * 3 ).fill( fill )
					}
				],
				outputActivation: { type: 'linear' }
			}
		}
	};

}

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceNodeMaterial', () => {

			test( 'updates matching weights without rebuilding', () => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const textures = material.neuralAppearanceData.latentTextures;
				const weightVector = material._outputUniforms.brdf.parameters.array[ material._outputUniforms.brdf.layers[ 0 ].weightsOffset ];
				const next = loader.parse( createManifest( 0.25 ) );

				expect( material.updateFromData( next ) ).toBe( true );
				expect( material.neuralAppearanceData.latentTextures ).toBe( textures );
				expect( material._outputUniforms.brdf.parameters.array[ material._outputUniforms.brdf.layers[ 0 ].weightsOffset ] ).toBe( weightVector );
				expect( weightVector.x ).not.toBe( 0 );

			} );

			test( 'rejects a different decoder layout', () => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const nextManifest = createManifest( 0 );
				nextManifest.outputs.brdf.layers = [
					{
						inputSize: 28,
						outputSize: 8,
						activation: 'relu',
						biases: new Array( 8 ).fill( 0 ),
						weights: new Array( 224 ).fill( 0 )
					},
					{
						inputSize: 8,
						outputSize: 3,
						activation: 'linear',
						biases: new Array( 3 ).fill( 0 ),
						weights: new Array( 24 ).fill( 0 )
					}
				];

				expect( material.updateFromData( loader.parse( nextManifest ) ) ).toBe( false );

			} );

			test( 'configures mask and blend opacity runtime state', () => {

				const loader = new NeuralAppearanceLoader();

				const maskManifest = createManifest( 0 );
				maskManifest.outputs.opacity = {
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
					mode: 'mask',
					alphaCutoff: 0.5
				};

				const maskMaterial = new NeuralAppearanceNodeMaterial( loader.parse( maskManifest ) );
				expect( maskMaterial.transparent ).toBe( false );
				expect( maskMaterial.alphaTest ).toBe( 0.5 );
				expect( maskMaterial.alphaTestNode ).toBeTruthy();
				expect( maskMaterial.opacityNode ).toBeTruthy();
				expect( maskMaterial.maskNode ).toBeTruthy();

				const blendManifest = createManifest( 0 );
				blendManifest.outputs.opacity = {
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
					mode: 'blend'
				};

				const blendMaterial = new NeuralAppearanceNodeMaterial( loader.parse( blendManifest ) );
				expect( blendMaterial.transparent ).toBe( true );
				expect( blendMaterial.depthWrite ).toBe( false );
				expect( blendMaterial.opacityNode ).toBeTruthy();
				expect( blendMaterial.alphaTestNode ).toBe( null );
				expect( blendMaterial.maskNode ).toBe( null );

				expect( maskMaterial.updateFromData( loader.parse( blendManifest ) ) ).toBe( false );

			} );

			test( 'builds debug shading graphs for decoder-frame visualization', () => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const debug = evaluateNeuralDebugShading( material );

				expect( material.debugView ).toBe( 'shaded' );
				expect( debug.viewNormal ).toBeTruthy();
				expect( debug.viewReflect ).toBeTruthy();
				expect( debug.viewIrradiance ).toBeTruthy();
				expect( debug.roughness ).toBeTruthy();

				material.debugView = 'roughness';
				expect( material.debugView ).toBe( 'roughness' );
				material.debugView = 'ibl';
				expect( material.debugView ).toBe( 'ibl' );

			} );

		} );

	} );

} );

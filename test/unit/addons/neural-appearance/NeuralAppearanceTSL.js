import { describe, test, expect } from 'vitest';
import { Vector4 } from 'three';
import {
	packLayerWeights,
	packLayerBiases,
	createOutputUniforms,
	isCompatibleNeuralAppearanceData,
	updateOutputUniforms,
	copyLatentTextureData
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTSL.js';

function fakeTexture( fill = 0 ) {

	const data = new Uint16Array( [ fill, 0, 0, 0 ] );

	return {
		image: { width: 1, height: 1, data },
		needsUpdate: false,
		dispose() {}
	};

}

function fakeData( weight = 0 ) {

	return {
		isNeuralAppearanceData: true,
		levels: 4,
		wrap: 'repeat',
		latentTextures: [ fakeTexture( weight ), fakeTexture( 0 ), fakeTexture( 0 ), fakeTexture( 0 ) ],
		outputs: {
			brdf: {
				inputSize: 28,
				rotation: { weights: new Array( 192 ).fill( weight ), inputSize: 16, outputSize: 12 },
				layers: [ {
					inputSize: 28,
					outputSize: 3,
					activation: 'linear',
					weights: new Array( 84 ).fill( weight ),
					biases: [ weight, 0, 0 ]
				} ],
				outputActivation: { type: 'linear' }
			},
			ibl: {
				inputSize: 22,
				layers: [ {
					inputSize: 22,
					outputSize: 4,
					activation: 'linear',
					weights: new Array( 22 * 4 ).fill( weight ),
					biases: [ 0, 0, 1, 0 ]
				} ],
				outputActivation: { type: 'linear' }
			},
			indirectRadiance: {
				inputSize: 22,
				layers: [ {
					inputSize: 22,
					outputSize: 3,
					activation: 'linear',
					weights: new Array( 22 * 3 ).fill( weight ),
					biases: [ 0, 0, 0 ]
				} ],
				outputActivation: { type: 'linear' }
			},
			indirectIrradiance: {
				inputSize: 22,
				layers: [ {
					inputSize: 22,
					outputSize: 3,
					activation: 'linear',
					weights: new Array( 22 * 3 ).fill( weight ),
					biases: [ 0, 0, 0 ]
				} ],
				outputActivation: { type: 'linear' }
			}
		}
	};

}

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceTSL', () => {

			test( 'packs layer weights into vec4 uniform array', () => {

				// 6 inputs (ceil(6/4) = 2 vec4s per output) and 2 outputs => 4 Vector4s
				const weights = [
					1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12
				];
				const packed = packLayerWeights( weights, 6, 2 );

				expect( packed.length ).toBe( 4 );
				expect( packed[ 0 ] instanceof Vector4 ).toBeTruthy();
				expect( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ] ).toEqual( [ 1, 2, 3, 4 ] );
				expect( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ] ).toEqual( [ 5, 6, 0, 0 ] );
				expect( [ packed[ 2 ].x, packed[ 2 ].y, packed[ 2 ].z, packed[ 2 ].w ] ).toEqual( [ 7, 8, 9, 10 ] );
				expect( [ packed[ 3 ].x, packed[ 3 ].y, packed[ 3 ].z, packed[ 3 ].w ] ).toEqual( [ 11, 12, 0, 0 ] );

			} );

			test( 'packs layer biases into vec4 uniform array', () => {

				const biases = [ 1, 2, 3, 4, 5 ];
				const packed = packLayerBiases( biases );

				expect( packed.length ).toBe( 2 );
				expect( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ] ).toEqual( [ 1, 2, 3, 4 ] );
				expect( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ] ).toEqual( [ 5, 0, 0, 0 ] );

			} );

			test( 'creates output uniforms for heads', () => {

				const outputs = {
					brdf: {
						rotation: { weights: new Array( 192 ).fill( 0 ), inputSize: 16, outputSize: 12 },
						layers: [ { weights: new Array( 84 ).fill( 0 ), biases: [ 0, 0, 0 ], inputSize: 28, outputSize: 3 } ]
					},
					ibl: {
						layers: [ { weights: new Array( 22 * 4 ).fill( 0 ), biases: new Array( 4 ).fill( 0 ), inputSize: 22, outputSize: 4 } ]
					},
					indirectRadiance: {
						layers: [ { weights: new Array( 22 * 3 ).fill( 0 ), biases: new Array( 3 ).fill( 0 ), inputSize: 22, outputSize: 3 } ]
					},
					indirectIrradiance: {
						layers: [ { weights: new Array( 22 * 3 ).fill( 0 ), biases: new Array( 3 ).fill( 0 ), inputSize: 22, outputSize: 3 } ]
					},
					emission: {
						layers: [ { weights: new Array( 48 ).fill( 0 ), biases: [ 0, 0, 0 ], inputSize: 16, outputSize: 3 } ]
					},
					opacity: {
						layers: [ { weights: new Array( 16 ).fill( 0 ), biases: [ 0 ], inputSize: 16, outputSize: 1 } ]
					}
				};

				const uniforms = createOutputUniforms( outputs );

				expect( uniforms.brdf ).toBeTruthy();
				expect( uniforms.brdf.parameters ).toBeTruthy();
				expect( uniforms.brdf.rotationWeightsOffset ).toBe( 0 );
				expect( uniforms.brdf.layers.length ).toBe( 1 );
				expect( uniforms.ibl ).toBeTruthy();
				expect( uniforms.indirectRadiance ).toBeTruthy();
				expect( uniforms.indirectIrradiance ).toBeTruthy();
				expect( uniforms.emission ).toBeTruthy();
				expect( uniforms.emission.layers.length ).toBe( 1 );
				expect( uniforms.opacity ).toBeTruthy();
				expect( uniforms.opacity.layers.length ).toBe( 1 );

			} );

			test( 'accepts compatible latent and decoder layouts', () => {

				expect( isCompatibleNeuralAppearanceData( fakeData( 0 ), fakeData( 1 ) ) ).toBeTruthy();

				const wider = fakeData( 0 );
				wider.latentTextures[ 0 ].image.width = 2;
				expect( isCompatibleNeuralAppearanceData( fakeData( 0 ), wider ) ).toBeFalsy();

				const hidden = fakeData( 0 );
				hidden.outputs.brdf.layers[ 0 ].outputSize = 8;
				expect( isCompatibleNeuralAppearanceData( fakeData( 0 ), hidden ) ).toBeFalsy();

			} );

			test( 'updates packed decoder uniforms in place', () => {

				const uniforms = createOutputUniforms( fakeData( 0 ).outputs );
				const first = uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].weightsOffset ];
				const bias = uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].biasesOffset ];

				updateOutputUniforms( uniforms, fakeData( 7 ).outputs );

				expect( uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].weightsOffset ] ).toBe( first );
				expect( first.x ).toBe( 7 );
				expect( uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].biasesOffset ] ).toBe( bias );
				expect( bias.x ).toBe( 7 );

			} );

			test( 'copies latent level data into existing textures', () => {

				const current = fakeData( 1 ).latentTextures;
				const next = fakeData( 9 ).latentTextures;

				copyLatentTextureData( current, next );

				expect( current[ 0 ].image.data[ 0 ] ).toBe( 9 );
				expect( current[ 0 ].needsUpdate ).toBe( true );

			} );

		} );

	} );

} );

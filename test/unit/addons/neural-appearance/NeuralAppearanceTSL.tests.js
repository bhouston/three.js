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

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceTSL', () => {

			QUnit.test( 'packs layer weights into vec4 uniform array', ( assert ) => {

				// 6 inputs (ceil(6/4) = 2 vec4s per output) and 2 outputs => 4 Vector4s
				const weights = [
					1, 2, 3, 4, 5, 6,
					7, 8, 9, 10, 11, 12
				];
				const packed = packLayerWeights( weights, 6, 2 );

				assert.strictEqual( packed.length, 4, 'packs into 4 Vector4s' );
				assert.ok( packed[ 0 ] instanceof Vector4, 'elements are Vector4 instances' );
				assert.deepEqual( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ], [ 1, 2, 3, 4 ], 'first vector contains first 4 weights' );
				assert.deepEqual( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ], [ 5, 6, 0, 0 ], 'second vector zero-padded' );
				assert.deepEqual( [ packed[ 2 ].x, packed[ 2 ].y, packed[ 2 ].z, packed[ 2 ].w ], [ 7, 8, 9, 10 ], 'third vector starts output 1' );
				assert.deepEqual( [ packed[ 3 ].x, packed[ 3 ].y, packed[ 3 ].z, packed[ 3 ].w ], [ 11, 12, 0, 0 ], 'fourth vector zero-padded' );

			} );

			QUnit.test( 'packs layer biases into vec4 uniform array', ( assert ) => {

				const biases = [ 1, 2, 3, 4, 5 ];
				const packed = packLayerBiases( biases );

				assert.strictEqual( packed.length, 2, 'packs 5 biases into 2 Vector4s' );
				assert.deepEqual( [ packed[ 0 ].x, packed[ 0 ].y, packed[ 0 ].z, packed[ 0 ].w ], [ 1, 2, 3, 4 ], 'first vector filled' );
				assert.deepEqual( [ packed[ 1 ].x, packed[ 1 ].y, packed[ 1 ].z, packed[ 1 ].w ], [ 5, 0, 0, 0 ], 'second vector zero-padded' );

			} );

			QUnit.test( 'creates output uniforms for heads', ( assert ) => {

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

				assert.ok( uniforms.brdf, 'creates brdf uniforms' );
				assert.ok( uniforms.brdf.parameters, 'creates packed parameter uniform node' );
				assert.strictEqual( uniforms.brdf.rotationWeightsOffset, 0, 'stores rotation offset' );
				assert.strictEqual( uniforms.brdf.layers.length, 1, 'creates 1 layer for brdf' );
				assert.ok( uniforms.ibl, 'creates IBL uniforms' );
				assert.ok( uniforms.indirectRadiance, 'creates radiance IBL uniforms' );
				assert.ok( uniforms.indirectIrradiance, 'creates irradiance IBL uniforms' );
				assert.ok( uniforms.emission, 'creates emission uniforms' );
				assert.strictEqual( uniforms.emission.layers.length, 1, 'creates 1 layer for emission' );
				assert.ok( uniforms.opacity, 'creates opacity uniforms' );
				assert.strictEqual( uniforms.opacity.layers.length, 1, 'creates 1 layer for opacity' );

			} );

			QUnit.test( 'accepts compatible latent and decoder layouts', ( assert ) => {

				assert.ok( isCompatibleNeuralAppearanceData( fakeData( 0 ), fakeData( 1 ) ), 'weight changes stay compatible' );

				const wider = fakeData( 0 );
				wider.latentTextures[ 0 ].image.width = 2;
				assert.notOk( isCompatibleNeuralAppearanceData( fakeData( 0 ), wider ), 'rejects a different latent size' );

				const hidden = fakeData( 0 );
				hidden.outputs.brdf.layers[ 0 ].outputSize = 8;
				assert.notOk( isCompatibleNeuralAppearanceData( fakeData( 0 ), hidden ), 'rejects a different decoder width' );

			} );

			QUnit.test( 'updates packed decoder uniforms in place', ( assert ) => {

				const uniforms = createOutputUniforms( fakeData( 0 ).outputs );
				const first = uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].weightsOffset ];
				const bias = uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].biasesOffset ];

				updateOutputUniforms( uniforms, fakeData( 7 ).outputs );

				assert.strictEqual( uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].weightsOffset ], first, 'keeps the same Vector4 instances' );
				assert.strictEqual( first.x, 7, 'writes new packed weights into the existing array' );
				assert.strictEqual( uniforms.brdf.parameters.array[ uniforms.brdf.layers[ 0 ].biasesOffset ], bias, 'keeps the same packed bias vector' );
				assert.strictEqual( bias.x, 7, 'writes new packed biases' );

			} );

			QUnit.test( 'copies latent level data into existing textures', ( assert ) => {

				const current = fakeData( 1 ).latentTextures;
				const next = fakeData( 9 ).latentTextures;

				copyLatentTextureData( current, next );

				assert.strictEqual( current[ 0 ].image.data[ 0 ], 9, 'overwrites destination level data' );
				assert.strictEqual( current[ 0 ].needsUpdate, true, 'marks the destination texture for upload' );

			} );

		} );

	} );

} );

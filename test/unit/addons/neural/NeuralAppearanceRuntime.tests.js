import { DataUtils } from 'three';
import {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs
} from '../../../../examples/jsm/neural/NeuralAppearanceRuntime.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceRuntime', () => {

			QUnit.test( 'evaluates exported half-float latents with runtime LOD selection', ( assert ) => {

				const outputWeights = new Array( 3 * 20 ).fill( 0 );
				outputWeights[ 0 ] = 1;
				outputWeights[ 20 + 1 ] = 1;
				outputWeights[ 40 + 2 ] = 1;
				const json = {
					latents: {
						textures: [
							{
								wrap: 'repeat',
								mipmaps: [
									{ width: 2, height: 2, data: [ 0.1, 0.2, 0.3, 0, 0.4, 0.5, 0.6, 0, 0.7, 0.8, 0.9, 0, 1, 1.1, 1.2, 0 ] },
									{ width: 1, height: 1, data: [ 0.75, 0.5, 0.25, 0 ] }
								]
							},
							{
								wrap: 'repeat',
								mipmaps: [
									{ width: 2, height: 2, data: new Array( 16 ).fill( 0 ) },
									{ width: 1, height: 1, data: [ 0, 0, 0, 0 ] }
								]
							}
						]
					},
					outputs: {
						brdf: {
							rotation: { weights: new Array( 8 * 12 ).fill( 0 ) },
							layers: [ {
								inputSize: 20,
								outputSize: 3,
								activation: 'linear',
								weights: outputWeights,
								biases: [ 0, 0, 0 ]
							} ],
							outputActivation: { type: 'linear' }
						}
					}
				};
				const direction = [ 0, 0, 1 ];
				const base = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: direction,
					wo: direction,
					duvDx: [ 0, 0 ],
					duvDy: [ 0, 0 ]
				} );
				const coarse = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: direction,
					wo: direction,
					duvDx: [ 1, 0 ],
					duvDy: [ 0, 1 ]
				} );
				const half = ( value ) => DataUtils.fromHalfFloat( DataUtils.toHalfFloat( value ) );

				assert.deepEqual( base, [ half( 0.1 ), half( 0.2 ), half( 0.3 ) ], 'samples the base mip at the requested UV after half-float conversion' );
				assert.deepEqual( coarse, [ half( 0.75 ), half( 0.5 ), half( 0.25 ) ], 'selects the coarse mip from the runtime UV footprint' );

				json.outputs.brdf.rotation.weights[ 5 * 8 ] = 10;
				json.outputs.brdf.layers[ 0 ].weights.fill( 0 );
				json.outputs.brdf.layers[ 0 ].weights[ 9 ] = 1;
				const learnedBitangent = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: [ 0, 1, 0 ],
					wo: direction,
					mip: 0
				} );

				assert.ok( Math.abs( learnedBitangent[ 0 ] - 1 ) < 1e-12, 'normalizes the learned bitangent before encoding directions' );

			} );

			QUnit.test( 'evaluates optional emission and opacity heads', ( assert ) => {

				const json = {
					latents: {
						textures: [
							{
								wrap: 'repeat',
								mipmaps: [ { width: 1, height: 1, data: [ 0.5, 0.5, 0.5, 0.5 ] } ]
							},
							{
								wrap: 'repeat',
								mipmaps: [ { width: 1, height: 1, data: [ 0.5, 0.5, 0.5, 0.5 ] } ]
							}
						]
					},
					outputs: {
						brdf: {
							rotation: { weights: new Array( 96 ).fill( 0 ) },
							layers: [ {
								inputSize: 20,
								outputSize: 3,
								activation: 'linear',
								weights: new Array( 60 ).fill( 0 ),
								biases: [ 0.1, 0.2, 0.3 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						emission: {
							layers: [ {
								inputSize: 8,
								outputSize: 3,
								activation: 'linear',
								weights: new Array( 24 ).fill( 0 ),
								biases: [ 1, 0.5, 0.25 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						opacity: {
							layers: [ {
								inputSize: 8,
								outputSize: 1,
								activation: 'linear',
								weights: new Array( 8 ).fill( 0 ),
								biases: [ 0 ]
							} ],
							outputActivation: { type: 'sigmoid' }
						}
					}
				};

				const outputs = evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					mip: 0
				} );

				assert.deepEqual( outputs.brdf, [ 0.1, 0.2, 0.3 ], 'evaluates brdf output' );
				assert.deepEqual( outputs.emission, [ 1, 0.5, 0.25 ], 'evaluates emission output' );
				assert.ok( Math.abs( outputs.opacity - 0.5 ) < 1e-6, 'evaluates sigmoid opacity output' );

			} );

		} );

	} );

} );

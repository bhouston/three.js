import { DataUtils } from 'three';
import {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateNeuralPrefilteredIBL
} from '../../../../examples/jsm/neural/NeuralAppearanceRuntime.js';

function createIBLOutput() {

	return {
		inputSize: 14,
		layers: [
			{
				inputSize: 14,
				outputSize: 4,
				activation: 'linear',
				weights: new Array( 14 * 4 ).fill( 0 ),
				biases: [ 0, 0, 1, 0 ]
			}
		],
		outputActivation: { type: 'linear' }
	};

}

function createIndirectOutput() {

	const weights = new Array( 14 * 3 ).fill( 0 );
	weights[ 11 ] = 1;
	weights[ 14 + 12 ] = 1;
	weights[ 28 + 13 ] = 1;

	return {
		inputSize: 14,
		layers: [
			{
				inputSize: 14,
				outputSize: 3,
				activation: 'linear',
				weights,
				biases: [ 0, 0, 0 ]
			}
		],
		outputActivation: { type: 'linear' }
	};

}

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
						},
						ibl: createIBLOutput(),
						indirect: createIndirectOutput()
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
						ibl: createIBLOutput(),
						indirect: createIndirectOutput(),
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
				assert.deepEqual( outputs.ibl.direction, [ 0, 0, 1 ], 'exposes IBL query direction' );
				assert.ok( Math.abs( outputs.ibl.roughness - 0.5 ) < 1e-6, 'exposes IBL query roughness' );
				assert.deepEqual( outputs.emission, [ 1, 0.5, 0.25 ], 'evaluates emission output' );
				assert.ok( Math.abs( outputs.opacity - 0.5 ) < 1e-6, 'evaluates sigmoid opacity output' );

				const prefiltered = evaluateNeuralPrefilteredIBL( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					mip: 0,
					iblIncoming: [ 2, 3, 4 ]
				} );

				assert.deepEqual( prefiltered, [ 2, 3, 4 ], 'maps incoming environment radiance through the indirect decoder' );

			} );

			QUnit.test( 'evaluates dual-level trilinear shader blending across fractional mip levels', ( assert ) => {

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
									{ width: 2, height: 2, data: [ 0.2, 0.4, 0.6, 0, 0.2, 0.4, 0.6, 0, 0.2, 0.4, 0.6, 0, 0.2, 0.4, 0.6, 0 ] },
									{ width: 1, height: 1, data: [ 0.8, 0.6, 0.4, 0 ] }
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
						},
						ibl: createIBLOutput(),
						indirect: createIndirectOutput(),
						emission: {
							layers: [ {
								inputSize: 8,
								outputSize: 3,
								activation: 'linear',
								weights: [
									1, 0, 0, 0, 0, 0, 0, 0,
									0, 1, 0, 0, 0, 0, 0, 0,
									0, 0, 1, 0, 0, 0, 0, 0
								],
								biases: [ 0, 0, 0 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						opacity: {
							layers: [ {
								inputSize: 8,
								outputSize: 1,
								activation: 'linear',
								weights: [ 1, 0, 0, 0, 0, 0, 0, 0 ],
								biases: [ 0 ]
							} ],
							outputActivation: { type: 'linear' }
						}
					}
				};

				const half = ( value ) => DataUtils.fromHalfFloat( DataUtils.toHalfFloat( value ) );
				const direction = [ 0, 0, 1 ];

				// Footprint where dx = sqrt(2) = 1.41421356 => log2(sqrt(2)) = 0.5 (halfway between mip 0 and mip 1)
				// Base mip 0 (latents [0.2, 0.4, 0.6]), coarse mip 1 (latents [0.8, 0.6, 0.4])
				const trilinearOutputs = evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.25, 0.25 ],
					wi: direction,
					wo: direction,
					duvDx: [ Math.SQRT2 / 2, 0 ],
					duvDy: [ 0, Math.SQRT2 / 2 ],
					lodMode: 'trilinear'
				} );

				const expectedBrdf = [
					half( 0.2 ) * 0.5 + half( 0.8 ) * 0.5,
					half( 0.4 ) * 0.5 + half( 0.6 ) * 0.5,
					half( 0.6 ) * 0.5 + half( 0.4 ) * 0.5
				];

				assert.ok( Math.abs( trilinearOutputs.brdf[ 0 ] - expectedBrdf[ 0 ] ) < 1e-6, 'trilinearly blends red brdf channel' );
				assert.ok( Math.abs( trilinearOutputs.brdf[ 1 ] - expectedBrdf[ 1 ] ) < 1e-6, 'trilinearly blends green brdf channel' );
				assert.ok( Math.abs( trilinearOutputs.brdf[ 2 ] - expectedBrdf[ 2 ] ) < 1e-6, 'trilinearly blends blue brdf channel' );
				assert.ok( Math.abs( trilinearOutputs.emission[ 0 ] - expectedBrdf[ 0 ] ) < 1e-6, 'trilinearly blends emission output' );
				assert.ok( Math.abs( trilinearOutputs.opacity - ( half( 0.2 ) * 0.5 + half( 0.8 ) * 0.5 ) ) < 1e-6, 'trilinearly blends opacity output' );

				const deterministicOutputs = evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.25, 0.25 ],
					wi: direction,
					wo: direction,
					duvDx: [ Math.SQRT2 / 2, 0 ],
					duvDy: [ 0, Math.SQRT2 / 2 ],
					lodMode: 'deterministic'
				} );

				// Deterministic round(0.5) selects mip 1
				assert.deepEqual( deterministicOutputs.brdf, [ half( 0.8 ), half( 0.6 ), half( 0.4 ) ], 'deterministic mode selects nearest integer mip without blending' );

			} );

			QUnit.test( 'evaluates IBL query and incoming-light indirect heads', ( assert ) => {

				const json = {
					latents: {
						textures: [
							{ wrap: 'repeat', mipmaps: [ { width: 1, height: 1, data: [ 0, 0, 0, 0 ] } ] },
							{ wrap: 'repeat', mipmaps: [ { width: 1, height: 1, data: [ 0, 0, 0, 0 ] } ] }
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
								biases: [ 0, 0, 0 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						ibl: createIBLOutput(),
						indirect: createIndirectOutput()
					}
				};
				const outputs = evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					mip: 0,
					iblIncoming: [ 0.2, 0.4, 0.6 ]
				} );

				assert.deepEqual( outputs.ibl.direction, [ 0, 0, 1 ], 'evaluates IBL query direction' );
				assert.ok( Math.abs( outputs.ibl.roughness - 0.5 ) < 1e-6, 'evaluates IBL query roughness' );
				assert.deepEqual( outputs.indirect, [ 0.2, 0.4, 0.6 ], 'passes incoming radiance through the indirect decoder' );
				assert.deepEqual( evaluateNeuralPrefilteredIBL( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					mip: 0,
					iblIncoming: [ 0.2, 0.4, 0.6 ]
				} ), [ 0.2, 0.4, 0.6 ], 'prefiltered IBL helper uses the indirect decoder' );

			} );

		} );

	} );

} );

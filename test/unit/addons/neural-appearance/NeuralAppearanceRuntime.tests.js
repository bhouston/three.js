import {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateNeuralPrefilteredIBL
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceRuntime.js';

function createIBLOutput() {

	return {
		inputSize: 22,
		layers: [
			{
				inputSize: 22,
				outputSize: 4,
				activation: 'linear',
				weights: new Array( 22 * 4 ).fill( 0 ),
				biases: [ 0, 0, 1, 0 ]
			}
		],
		outputActivation: { type: 'linear' }
	};

}

function createIndirectProbeOutput( passthrough = false ) {

	const weights = new Array( 22 * 3 ).fill( 0 );
	if ( passthrough ) {

		weights[ 19 ] = 1;
		weights[ 22 + 20 ] = 1;
		weights[ 44 + 21 ] = 1;

	}

	return {
		inputSize: 22,
		layers: [
			{
				inputSize: 22,
				outputSize: 3,
				activation: 'linear',
				weights,
				biases: [ 0, 0, 0 ]
			}
		],
		outputActivation: { type: 'linear' }
	};

}

function createLevels( fillPerLevel ) {

	return fillPerLevel.map( ( fill ) => ( {
		width: 1,
		height: 1,
		channels: 4,
		wrap: 'repeat',
		data: Array.isArray( fill ) ? fill : [ fill, fill, fill, fill ]
	} ) );

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceRuntime', () => {

			QUnit.test( 'evaluates the multiresolution grid + decoder', ( assert ) => {

				// 16 latents identity-projected into the first 3 decoder
				// output channels (weights row j selects latent channel j).
				const outputWeights = new Array( 3 * 28 ).fill( 0 );
				outputWeights[ 0 ] = 1;
				outputWeights[ 28 + 1 ] = 1;
				outputWeights[ 56 + 2 ] = 1;
				const json = {
					latents: {
						levels: createLevels( [
							[ 0.1, 0.2, 0.3, 0 ],
							[ 0.4, 0.5, 0.6, 0 ],
							[ 0.7, 0.8, 0.9, 0 ],
							[ 0, 0, 0, 0 ]
						] )
					},
					outputs: {
						brdf: {
							rotation: { weights: new Array( 16 * 12 ).fill( 0 ) },
							layers: [ {
								inputSize: 28,
								outputSize: 3,
								activation: 'linear',
								weights: outputWeights,
								biases: [ 0, 0, 0 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						ibl: createIBLOutput(),
						indirectRadiance: createIndirectProbeOutput(),
						indirectIrradiance: createIndirectProbeOutput()
					}
				};
				const direction = [ 0, 0, 1 ];
				const result = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: direction,
					wo: direction
				} );

				assert.ok( Math.abs( result[ 0 ] - 0.1 ) < 1e-9 && Math.abs( result[ 1 ] - 0.2 ) < 1e-9 && Math.abs( result[ 2 ] - 0.3 ) < 1e-9, 'concatenates level 0 latents into the decoder input in order' );

				json.outputs.brdf.rotation.weights[ 5 * 16 ] = 10;
				json.outputs.brdf.layers[ 0 ].weights.fill( 0 );
				json.outputs.brdf.layers[ 0 ].weights[ 17 ] = 1;
				const learnedBitangent = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: [ 0, 1, 0 ],
					wo: direction
				} );

				assert.ok( Math.abs( learnedBitangent[ 0 ] - 1 ) < 1e-12, 'normalizes the learned bitangent before encoding directions' );

			} );

			QUnit.test( 'evaluates optional emission and opacity heads', ( assert ) => {

				const json = {
					latents: {
						levels: createLevels( [ 0.5, 0.5, 0.5, 0.5 ] )
					},
					outputs: {
						brdf: {
							rotation: { weights: new Array( 192 ).fill( 0 ) },
							layers: [ {
								inputSize: 28,
								outputSize: 3,
								activation: 'linear',
								weights: new Array( 84 ).fill( 0 ),
								biases: [ 0.1, 0.2, 0.3 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						ibl: createIBLOutput(),
						indirectRadiance: createIndirectProbeOutput( true ),
						indirectIrradiance: createIndirectProbeOutput(),
						emission: {
							layers: [ {
								inputSize: 16,
								outputSize: 3,
								activation: 'linear',
								weights: new Array( 48 ).fill( 0 ),
								biases: [ 1, 0.5, 0.25 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						opacity: {
							layers: [ {
								inputSize: 16,
								outputSize: 1,
								activation: 'linear',
								weights: new Array( 16 ).fill( 0 ),
								biases: [ 0 ]
							} ],
							outputActivation: { type: 'sigmoid' }
						}
					}
				};

				const outputs = evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ]
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
					iblIncoming: [ 2, 3, 4 ]
				} );

				assert.deepEqual( prefiltered, [ 2, 3, 4 ], 'maps incoming environment radiance through the radiance IBL head' );

			} );

			QUnit.test( 'evaluates IBL query and incoming-light indirect heads', ( assert ) => {

				const json = {
					latents: {
						levels: createLevels( [ 0, 0, 0, 0 ] )
					},
					outputs: {
						brdf: {
							rotation: { weights: new Array( 192 ).fill( 0 ) },
							layers: [ {
								inputSize: 28,
								outputSize: 3,
								activation: 'linear',
								weights: new Array( 84 ).fill( 0 ),
								biases: [ 0, 0, 0 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						ibl: createIBLOutput(),
						indirectRadiance: createIndirectProbeOutput( true ),
						indirectIrradiance: createIndirectProbeOutput()
					}
				};
				const outputs = evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					iblIncoming: [ 0.2, 0.4, 0.6 ]
				} );

				assert.deepEqual( outputs.ibl.direction, [ 0, 0, 1 ], 'evaluates IBL query direction' );
				assert.ok( Math.abs( outputs.ibl.roughness - 0.5 ) < 1e-6, 'evaluates IBL query roughness' );
				assert.deepEqual( outputs.indirectRadiance, [ 0.2, 0.4, 0.6 ], 'passes incoming radiance through the radiance IBL head' );
				assert.deepEqual( outputs.indirect, [ 0.2, 0.4, 0.6 ], 'sums isolate heads for full indirect lighting' );
				assert.deepEqual( evaluateNeuralPrefilteredIBL( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					iblIncoming: [ 0.2, 0.4, 0.6 ]
				} ), [ 0.2, 0.4, 0.6 ], 'prefiltered IBL helper uses the summed isolate heads' );

				json.outputs.indirectRadiance = createIndirectProbeOutput();
				json.outputs.indirectIrradiance = createIndirectProbeOutput( true );
				assert.deepEqual( evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					iblIncoming: [ 0.2, 0.4, 0.6 ],
					iblIrradiance: [ 0.7, 0.8, 0.9 ]
				} ).indirectIrradiance, [ 0.7, 0.8, 0.9 ], 'passes incoming irradiance through the irradiance IBL head' );

			} );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
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

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceRuntime', () => {

			test( 'evaluates the multiresolution grid + decoder', () => {

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

				expect( Math.abs( result[ 0 ] - 0.1 ) < 1e-9 && Math.abs( result[ 1 ] - 0.2 ) < 1e-9 && Math.abs( result[ 2 ] - 0.3 ) < 1e-9 ).toBeTruthy();

				json.outputs.brdf.rotation.weights[ 5 * 16 ] = 10;
				json.outputs.brdf.layers[ 0 ].weights.fill( 0 );
				json.outputs.brdf.layers[ 0 ].weights[ 17 ] = 1;
				const learnedBitangent = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: [ 0, 1, 0 ],
					wo: direction
				} );

				expect( Math.abs( learnedBitangent[ 0 ] - 1 ) < 1e-12 ).toBeTruthy();

			} );

			test( 'evaluates optional emission and opacity heads', () => {

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

				expect( outputs.brdf ).toEqual( [ 0.1, 0.2, 0.3 ] );
				expect( outputs.ibl.direction ).toEqual( [ 0, 0, 1 ] );
				expect( Math.abs( outputs.ibl.roughness - 0.5 ) < 1e-6 ).toBeTruthy();
				expect( outputs.emission ).toEqual( [ 1, 0.5, 0.25 ] );
				expect( Math.abs( outputs.opacity - 0.5 ) < 1e-6 ).toBeTruthy();

				const prefiltered = evaluateNeuralPrefilteredIBL( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					iblIncoming: [ 2, 3, 4 ]
				} );

				expect( prefiltered ).toEqual( [ 2, 3, 4 ] );

			} );

			test( 'evaluates IBL query and incoming-light indirect heads', () => {

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

				expect( outputs.ibl.direction ).toEqual( [ 0, 0, 1 ] );
				expect( Math.abs( outputs.ibl.roughness - 0.5 ) < 1e-6 ).toBeTruthy();
				expect( outputs.indirectRadiance ).toEqual( [ 0.2, 0.4, 0.6 ] );
				expect( outputs.indirect ).toEqual( [ 0.2, 0.4, 0.6 ] );
				expect( evaluateNeuralPrefilteredIBL( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					iblIncoming: [ 0.2, 0.4, 0.6 ]
				} ) ).toEqual( [ 0.2, 0.4, 0.6 ] );

				json.outputs.indirectRadiance = createIndirectProbeOutput();
				json.outputs.indirectIrradiance = createIndirectProbeOutput( true );
				expect( evaluateNeuralAppearanceOutputs( json, {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					iblIncoming: [ 0.2, 0.4, 0.6 ],
					iblIrradiance: [ 0.7, 0.8, 0.9 ]
				} ).indirectIrradiance ).toEqual( [ 0.7, 0.8, 0.9 ] );

			} );

		} );

	} );

} );

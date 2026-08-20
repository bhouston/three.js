import { describe, test, expect } from 'vitest';
import {
	evaluateRuntimeValidation,
	createDifferenceMetric,
	accumulateDifferenceMetric,
	finalizeDifferenceMetric,
	createAngularBins,
	accumulateAngularError,
	finalizeAngularBins
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceValidator.js';

function createLatentLevel( fill ) {

	return { width: 1, height: 1, channels: 4, wrap: 'repeat', data: new Array( 4 ).fill( fill ) };

}

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceValidator', () => {

			test( 'accumulates difference and angular metrics', () => {

				const metric = createDifferenceMetric();
				accumulateDifferenceMetric( metric, [ 1, 2, 3 ], [ 1.1, 1.9, 3.2 ] );

				const finalized = finalizeDifferenceMetric( metric );
				expect( finalized.sampleCount ).toBe( 1 );
				expect( Math.abs( finalized.meanAbsoluteDifference - ( 0.1 + 0.1 + 0.2 ) / 3 ) < 1e-6 ).toBeTruthy();

				const bins = createAngularBins();
				accumulateAngularError( bins, 0.5, [ 1, 1, 1 ], [ 0.9, 1.1, 1.0 ] );

				const finalizedBins = finalizeAngularBins( bins );
				expect( finalizedBins[ 2 ].count ).toBe( 1 );
				expect( Math.abs( finalizedBins[ 2 ].meanAbsoluteError - ( 0.1 + 0.1 + 0 ) / 3 ) < 1e-6 ).toBeTruthy();

			} );

			test( 'evaluates complete runtime validation metrics on synthetic samples', () => {

				const json = {
					latents: {
						levels: [
							createLatentLevel( 0 ),
							createLatentLevel( 0 ),
							createLatentLevel( 0 ),
							createLatentLevel( 0 )
						]
					},
					outputs: {
						brdf: {
							rotation: { weights: new Array( 192 ).fill( 0 ) },
							layers: [ {
								inputSize: 28,
								outputSize: 3,
								activation: 'linear',
								weights: new Array( 84 ).fill( 0 ),
								biases: [ 0.5, 0.5, 0.5 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						ibl: {
							inputSize: 22,
							layers: [ {
								inputSize: 22,
								outputSize: 4,
								activation: 'linear',
								weights: new Array( 22 * 4 ).fill( 0 ),
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
								weights: new Array( 22 * 3 ).fill( 0 ),
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
								weights: new Array( 22 * 3 ).fill( 0 ),
								biases: [ 0, 0, 0 ]
							} ],
							outputActivation: { type: 'linear' }
						}
					}
				};

				const samples = [
					{
						uv: [ 0.5, 0.5 ],
						wi: [ 0, 0, 1 ],
						wo: [ 0, 0, 1 ],
						target: [ 0.5, 0.5, 0.5 ],
						directTarget: [ 0.5, 0.5, 0.5 ],
						weight: 1
					}
				];

				const validation = evaluateRuntimeValidation( json, samples, 4 );

				expect( validation.sampleCount ).toBe( 1 );
				expect( validation.levels ).toBe( 4 );
				expect( Number.isFinite( validation.loss ) ).toBeTruthy();
				expect( Number.isFinite( validation.directLoss ) ).toBeTruthy();
				expect( validation.preview.samples.length ).toBe( 1 );
				expect( Number.isFinite( validation.reciprocity.meanAbsoluteDifference ) ).toBeTruthy();
				expect( Number.isFinite( validation.angularSmoothness.meanAbsoluteDifference ) ).toBeTruthy();
				expect( Number.isFinite( validation.whiteFurnace.meanAbsoluteDifference ) ).toBeTruthy();
				expect( Number.isFinite( validation.prefilteredIBL.meanAbsoluteDifference ) ).toBeTruthy();
				expect( validation.framePriors.sampleCount ).toBe( 1 );

			} );

		} );

	} );

} );

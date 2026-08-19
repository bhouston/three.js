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

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceValidator', () => {

			QUnit.test( 'accumulates difference and angular metrics', ( assert ) => {

				const metric = createDifferenceMetric();
				accumulateDifferenceMetric( metric, [ 1, 2, 3 ], [ 1.1, 1.9, 3.2 ] );

				const finalized = finalizeDifferenceMetric( metric );
				assert.strictEqual( finalized.sampleCount, 1, 'records 1 sample' );
				assert.ok( Math.abs( finalized.meanAbsoluteDifference - ( 0.1 + 0.1 + 0.2 ) / 3 ) < 1e-6, 'computes mean absolute difference' );

				const bins = createAngularBins();
				accumulateAngularError( bins, 0.5, [ 1, 1, 1 ], [ 0.9, 1.1, 1.0 ] );

				const finalizedBins = finalizeAngularBins( bins );
				assert.strictEqual( finalizedBins[ 2 ].count, 1, 'places cosine 0.5 in the high-angle bin' );
				assert.ok( Math.abs( finalizedBins[ 2 ].meanAbsoluteError - ( 0.1 + 0.1 + 0 ) / 3 ) < 1e-6, 'computes bin mean error' );

			} );

			QUnit.test( 'evaluates complete runtime validation metrics on synthetic samples', ( assert ) => {

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

				assert.strictEqual( validation.sampleCount, 1, 'evaluates sample count' );
				assert.strictEqual( validation.levels, 4, 'reports the number of latent grid levels' );
				assert.ok( Number.isFinite( validation.loss ), 'computes finite loss' );
				assert.ok( Number.isFinite( validation.directLoss ), 'computes finite direct loss' );
				assert.strictEqual( validation.preview.samples.length, 1, 'builds preview' );
				assert.ok( Number.isFinite( validation.reciprocity.meanAbsoluteDifference ), 'computes reciprocity metric' );
				assert.ok( Number.isFinite( validation.angularSmoothness.meanAbsoluteDifference ), 'computes angular smoothness metric' );
				assert.ok( Number.isFinite( validation.whiteFurnace.meanAbsoluteDifference ), 'computes white-furnace metric' );
				assert.ok( Number.isFinite( validation.prefilteredIBL.meanAbsoluteDifference ), 'computes prefiltered IBL metric' );
				assert.strictEqual( validation.framePriors.sampleCount, 1, 'computes frame prior metrics' );

			} );

		} );

	} );

} );

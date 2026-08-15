import {
	evaluateRuntimeValidation,
	createDifferenceMetric,
	accumulateDifferenceMetric,
	finalizeDifferenceMetric,
	createAngularBins,
	accumulateAngularError,
	finalizeAngularBins
} from '../../../../examples/jsm/neural/NeuralAppearanceValidator.js';

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
						textures: [
							{ wrap: 'repeat', mipmaps: [
								{ width: 2, height: 2, data: new Array( 16 ).fill( 0 ) },
								{ width: 1, height: 1, data: [ 0, 0, 0, 0 ] }
							] },
							{ wrap: 'repeat', mipmaps: [
								{ width: 2, height: 2, data: new Array( 16 ).fill( 0 ) },
								{ width: 1, height: 1, data: [ 0, 0, 0, 0 ] }
							] }
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
								biases: [ 0.5, 0.5, 0.5 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						ibl: {
							inputSize: 14,
							layers: [ {
								inputSize: 14,
								outputSize: 4,
								activation: 'linear',
								weights: new Array( 14 * 4 ).fill( 0 ),
								biases: [ 0, 0, 1, 0 ]
							} ],
							outputActivation: { type: 'linear' }
						},
						indirect: {
							inputSize: 17,
							layers: [ {
								inputSize: 17,
								outputSize: 3,
								activation: 'linear',
								weights: new Array( 17 * 3 ).fill( 0 ),
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
						mip: 0,
						target: [ 0.5, 0.5, 0.5 ],
						directTarget: [ 0.5, 0.5, 0.5 ],
						weight: 1
					}
				];

				const validation = evaluateRuntimeValidation( json, samples, 4 );

				assert.strictEqual( validation.sampleCount, 1, 'evaluates sample count' );
				assert.ok( Number.isFinite( validation.loss ), 'computes finite loss' );
				assert.strictEqual( validation.preview.samples.length, 1, 'builds preview' );
				assert.ok( Number.isFinite( validation.reciprocity.meanAbsoluteDifference ), 'computes reciprocity metric' );
				assert.ok( Number.isFinite( validation.angularSmoothness.meanAbsoluteDifference ), 'computes angular smoothness metric' );
				assert.ok( Number.isFinite( validation.whiteFurnace.meanAbsoluteDifference ), 'computes white-furnace metric' );
				assert.ok( Number.isFinite( validation.prefilteredIBL.meanAbsoluteDifference ), 'computes prefiltered IBL metric' );
				assert.strictEqual( validation.mipConsistency.sampleCount, 1, 'computes adjacent-mip consistency metric' );
				assert.strictEqual( validation.framePriors.sampleCount, 1, 'computes frame prior metrics' );

			} );

		} );

	} );

} );

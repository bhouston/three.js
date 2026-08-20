import { describe, expect, it } from 'vitest';
import { createModel } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';
import { createNeuralAppearanceManifest } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceManifest.js';
import {
	evaluateRuntimeValidation,
	createDifferenceMetric,
	accumulateDifferenceMetric,
	finalizeDifferenceMetric,
	createAngularBins,
	accumulateAngularError,
	finalizeAngularBins
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceValidator.js';

// Builds a tiny, real manifest via the real encoder (createModel +
// createNeuralAppearanceManifest) rather than hand-constructing fake manifest
// JSON. `levels` is intentionally left at its default: LATENT_CHANNELS in
// NeuralAppearanceFormat.js is `LEVELS * CHANNELS_PER_LEVEL`, a fixed module
// constant independent of any `levels` option passed here - overriding
// `levels` without also changing that constant would size the decoder input
// (which is derived from LATENT_CHANNELS) inconsistently with the actual
// concatenated latent length, breaking the forward pass. baseResolution/
// targetResolution/hiddenSize are shrunk instead, which is safe.
//
// random() = 0.5 maps to a weight of exactly 0 in every layer (see
// NeuralMLP.js: `weights[j] = (random() * 2 - 1) * scale`), so every head's
// forward pass collapses to "bias only", making every prediction exactly
// deterministic and easy to reason about by hand.
function createTinyManifest() {

	const random = () => 0.5;
	const model = createModel( {
		hiddenSize: 4,
		baseResolution: 2,
		targetResolution: 4,
		outputFeatures: {}
	}, random );

	return createNeuralAppearanceManifest( model, {
		name: 'validator-test',
		outputActivation: { type: 'linear' }
	} );

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceValidator', () => {

	describe( 'createDifferenceMetric / accumulateDifferenceMetric / finalizeDifferenceMetric', () => {

		it( 'creates a fresh metric at all zeros', () => {

			expect( createDifferenceMetric() ).toEqual( {
				sampleCount: 0,
				absoluteDifference: 0,
				magnitude: 0,
				maxChannelDifference: 0
			} );

		} );

		it( 'accumulates one (a, b) pair exactly as hand-computed', () => {

			const metric = createDifferenceMetric();
			accumulateDifferenceMetric( metric, [ 1, 2, 3 ], [ 0, 2, 5 ] );

			// absoluteDifference: |1-0| + |2-2| + |3-5| = 1 + 0 + 2 = 3
			// magnitude: 0.5*(1+0) + 0.5*(2+2) + 0.5*(3+5) = 0.5 + 2 + 4 = 6.5
			// maxChannelDifference: max(1, 0, 2) = 2
			expect( metric.sampleCount ).toBe( 1 );
			expect( metric.absoluteDifference ).toBeCloseTo( 3, 10 );
			expect( metric.magnitude ).toBeCloseTo( 6.5, 10 );
			expect( metric.maxChannelDifference ).toBeCloseTo( 2, 10 );

		} );

		it( 'is a no-op (does not increment sampleCount) when a contains a non-finite value', () => {

			const metric = createDifferenceMetric();
			accumulateDifferenceMetric( metric, [ NaN, 2, 3 ], [ 0, 2, 5 ] );

			expect( metric.sampleCount ).toBe( 0 );
			expect( metric.absoluteDifference ).toBe( 0 );

		} );

		it( 'is a no-op (does not increment sampleCount) when b contains a non-finite value', () => {

			const metric = createDifferenceMetric();
			accumulateDifferenceMetric( metric, [ 1, 2, 3 ], [ 0, Infinity, 5 ] );

			expect( metric.sampleCount ).toBe( 0 );
			expect( metric.absoluteDifference ).toBe( 0 );

		} );

		it( 'finalizes a zero-sample metric to null derived fields, not NaN/0', () => {

			const finalized = finalizeDifferenceMetric( createDifferenceMetric() );

			expect( finalized.sampleCount ).toBe( 0 );
			expect( finalized.meanAbsoluteDifference ).toBeNull();
			expect( finalized.relativeAbsoluteDifference ).toBeNull();
			expect( finalized.maxChannelDifference ).toBeNull();

		} );

		it( 'finalizes the hand-computed accumulation to the expected numbers', () => {

			const metric = createDifferenceMetric();
			accumulateDifferenceMetric( metric, [ 1, 2, 3 ], [ 0, 2, 5 ] );

			const finalized = finalizeDifferenceMetric( metric );

			expect( finalized.sampleCount ).toBe( 1 );
			expect( finalized.meanAbsoluteDifference ).toBeCloseTo( 3 / 3, 10 ); // absoluteDifference / (sampleCount * 3)
			expect( finalized.relativeAbsoluteDifference ).toBeCloseTo( 3 / 6.5, 10 ); // absoluteDifference / magnitude
			expect( finalized.maxChannelDifference ).toBeCloseTo( 2, 10 );

		} );

		it( 'relativeAbsoluteDifference is null when magnitude is 0, even with sampleCount > 0', () => {

			const metric = createDifferenceMetric();
			accumulateDifferenceMetric( metric, [ 0, 0, 0 ], [ 0, 0, 0 ] ); // two identical zero vectors

			expect( metric.sampleCount ).toBe( 1 );
			expect( metric.magnitude ).toBe( 0 );

			const finalized = finalizeDifferenceMetric( metric );

			expect( finalized.meanAbsoluteDifference ).toBe( 0 ); // sampleCount > 0, so 0, not null
			expect( finalized.relativeAbsoluteDifference ).toBeNull(); // guarded on magnitude > 0, not sampleCount
			expect( finalized.maxChannelDifference ).toBe( 0 );

		} );

	} );

	describe( 'createAngularBins / accumulateAngularError / finalizeAngularBins', () => {

		it( 'creates three bins with the grazing/mid/high cosine-of-normal ranges', () => {

			expect( createAngularBins() ).toEqual( [
				{ min: 0, max: 0.05, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 },
				{ min: 0.05, max: 0.2, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 },
				{ min: 0.2, max: 1, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 }
			] );

		} );

		it( 'routes an interior grazing-bin cosine into bin 0', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0.02, [ 0, 0, 0 ], [ 0, 0, 0 ] );

			expect( bins[ 0 ].count ).toBe( 1 );
			expect( bins[ 1 ].count ).toBe( 0 );
			expect( bins[ 2 ].count ).toBe( 0 );

		} );

		it( 'routes an interior mid-bin cosine into bin 1', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0.1, [ 0, 0, 0 ], [ 0, 0, 0 ] );

			expect( bins[ 0 ].count ).toBe( 0 );
			expect( bins[ 1 ].count ).toBe( 1 );
			expect( bins[ 2 ].count ).toBe( 0 );

		} );

		it( 'routes an interior high-bin cosine into bin 2', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0.5, [ 0, 0, 0 ], [ 0, 0, 0 ] );

			expect( bins[ 0 ].count ).toBe( 0 );
			expect( bins[ 1 ].count ).toBe( 0 );
			expect( bins[ 2 ].count ).toBe( 1 );

		} );

		it( 'the lower boundary of a bin (min) is inclusive to that bin: cosine = 0 lands in bin 0', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0, [ 0, 0, 0 ], [ 0, 0, 0 ] );

			expect( bins[ 0 ].count ).toBe( 1 );

		} );

		it( 'the upper boundary of a bin (max) is exclusive: cosine = 0.05 lands in bin 1, not bin 0', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0.05, [ 0, 0, 0 ], [ 0, 0, 0 ] );

			expect( bins[ 0 ].count ).toBe( 0 );
			expect( bins[ 1 ].count ).toBe( 1 );

		} );

		it( 'the upper boundary of bin 1 (0.2) is exclusive: it lands in bin 2, not bin 1', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0.2, [ 0, 0, 0 ], [ 0, 0, 0 ] );

			expect( bins[ 1 ].count ).toBe( 0 );
			expect( bins[ 2 ].count ).toBe( 1 );

		} );

		it( 'cosine = 1 (the extreme top of the range) lands in the last bin via its "last bin" fallback', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 1, [ 0, 0, 0 ], [ 0, 0, 0 ] );

			expect( bins[ 2 ].count ).toBe( 1 );

		} );

		it( 'is a no-op (does not increment count) when target contains a non-finite value', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0.5, [ 0, 0, 0 ], [ NaN, 0, 0 ] );

			expect( bins.every( ( bin ) => bin.count === 0 ) ).toBe( true );

		} );

		it( 'finalizes untouched (count = 0) bins to null derived fields, not NaN/0', () => {

			const finalized = finalizeAngularBins( createAngularBins() );

			for ( const bin of finalized ) {

				expect( bin.count ).toBe( 0 );
				expect( bin.meanAbsoluteError ).toBeNull();
				expect( bin.relativeAbsoluteError ).toBeNull();
				expect( bin.maxChannelError ).toBeNull();

			}

		} );

		it( 'finalizes a bin that received a sample to hand-computed numbers', () => {

			const bins = createAngularBins();
			accumulateAngularError( bins, 0.5, [ 1, 0, 0 ], [ 0, 0, 0.5 ] ); // prediction, target

			// per-channel error: |1-0| = 1, |0-0| = 0, |0-0.5| = 0.5
			// absoluteError = 1.5, targetMagnitude = |0|+|0|+|0.5| = 0.5, maxChannelError = 1
			const finalized = finalizeAngularBins( bins );
			const bin = finalized[ 2 ];

			expect( bin.count ).toBe( 1 );
			expect( bin.meanAbsoluteError ).toBeCloseTo( 1.5 / 3, 10 );
			expect( bin.relativeAbsoluteError ).toBeCloseTo( 1.5 / 0.5, 10 );
			expect( bin.maxChannelError ).toBeCloseTo( 1, 10 );

		} );

	} );

	describe( 'evaluateRuntimeValidation', () => {

		it( 'handles an empty sample list without throwing, reporting zero samples and zero loss', () => {

			const manifest = createTinyManifest();
			const result = evaluateRuntimeValidation( manifest, [], 0 );

			expect( result.sampleCount ).toBe( 0 );
			expect( result.loss ).toBe( 0 );
			expect( result.brdfLoss ).toBe( 0 );
			expect( result.directLoss ).toBe( 0 );

		} );

		it( 'round-trips a single well-formed sample without throwing and produces a finite, non-negative loss', () => {

			const manifest = createTinyManifest();
			const samples = [ {
				uv: [ 0.5, 0.5 ],
				wi: [ 0, 0, 1 ],
				wo: [ 0, 0, 1 ],
				target: [ 0.2, 0.3, 0.4 ],
				weight: 1
			} ];

			const result = evaluateRuntimeValidation( manifest, samples, 1 );

			expect( result.sampleCount ).toBe( 1 );
			expect( Number.isFinite( result.loss ) ).toBe( true );
			expect( result.loss ).toBeGreaterThanOrEqual( 0 );
			expect( Number.isFinite( result.brdfLoss ) ).toBe( true );
			expect( result.brdfLoss ).toBeGreaterThanOrEqual( 0 );

		} );

		it( 'a weight: 0 sample is still processed for reciprocity/angularSmoothness but contributes nothing to brdfLoss', () => {

			const manifest = createTinyManifest();
			const samples = [ {
				uv: [ 0.5, 0.5 ],
				wi: [ 0, 0, 1 ],
				wo: [ 0, 0, 1 ],
				target: [ 0.2, 0.3, 0.4 ],
				weight: 0
			} ];

			const result = evaluateRuntimeValidation( manifest, samples, 0 );

			// the brdfLoss accumulation loop is gated on `sampleWeight > 0`
			expect( result.brdfLoss ).toBe( 0 );
			expect( result.loss ).toBe( 0 );

			// reciprocity/angularSmoothness accumulation is not gated on weight
			expect( result.reciprocity.sampleCount ).toBe( 1 );
			expect( result.angularSmoothness.sampleCount ).toBe( 2 ); // one perturbed-wi + one perturbed-wo accumulation

		} );

		it( 'throws when a sample target contains a non-finite value while weight > 0', () => {

			const manifest = createTinyManifest();
			const samples = [ {
				uv: [ 0.5, 0.5 ],
				wi: [ 0, 0, 1 ],
				wo: [ 0, 0, 1 ],
				target: [ NaN, 0.3, 0.4 ],
				weight: 1
			} ];

			expect( () => evaluateRuntimeValidation( manifest, samples, 0 ) ).toThrow(
				'Runtime validation produced non-finite values.'
			);

		} );

		it( 'leaves emissionLoss/opacityLoss/iblLoss (and its sub-losses) null when nothing in the batch exercises them', () => {

			const manifest = createTinyManifest();

			// the tiny manifest is built with outputFeatures: {}, so it has no
			// emission/opacity heads at all; and the samples below supply no
			// ibl target fields (iblDirection/iblRoughness/iblIndirect*), so
			// the ibl-loss-specific counters stay at zero too.
			expect( manifest.outputs.emission ).toBeUndefined();
			expect( manifest.outputs.opacity ).toBeUndefined();

			const samples = [ {
				uv: [ 0.5, 0.5 ],
				wi: [ 0, 0, 1 ],
				wo: [ 0, 0, 1 ],
				target: [ 0.2, 0.3, 0.4 ],
				weight: 1
			} ];

			const result = evaluateRuntimeValidation( manifest, samples, 0 );

			expect( result.emissionLoss ).toBeNull();
			expect( result.opacityLoss ).toBeNull();
			expect( result.iblLoss ).toBeNull();
			expect( result.iblQueryLoss ).toBeNull();
			expect( result.iblIndirectLoss ).toBeNull();

		} );

	} );

} );

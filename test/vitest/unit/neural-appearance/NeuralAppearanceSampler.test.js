import { describe, expect, it } from 'vitest';
import {
	assignTeacherTargets,
	normalizeDirectLightingTargets
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceSampler.js';

// Minimal fake teachers - only the shape assignTeacherTargets/normalizeDirectLightingTargets
// actually consume: either an `evaluateBatch(samples)` batch entry point, or a
// per-sample async `evaluate(sample)` fallback. No GPU/renderer involved.
function createBatchTeacher( fn ) {

	return {
		evaluateBatch: async ( samples ) => samples.map( fn )
	};

}

function createPerSampleTeacher( fn ) {

	return {
		evaluate: async ( sample ) => fn( sample )
	};

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceSampler', () => {

	describe( 'assignTeacherTargets', () => {

		it( 'uses evaluateBatch when the teacher provides it, assigning target per sample in order', async () => {

			const teacher = createBatchTeacher( ( sample, i ) => [ i, i + 1, i + 2 ] );
			const samples = [ { uv: [ 0, 0 ] }, { uv: [ 1, 1 ] }, { uv: [ 2, 2 ] } ];

			await assignTeacherTargets( samples, teacher );

			expect( samples[ 0 ].target ).toEqual( [ 0, 1, 2 ] );
			expect( samples[ 1 ].target ).toEqual( [ 1, 2, 3 ] );
			expect( samples[ 2 ].target ).toEqual( [ 2, 3, 4 ] );

		} );

		it( 'falls back to per-sample evaluate() when the teacher has no evaluateBatch', async () => {

			const teacher = createPerSampleTeacher( ( sample ) => [ sample.uv[ 0 ] * 10, sample.uv[ 1 ] * 10, 0 ] );
			const samples = [ { uv: [ 0.1, 0.2 ] }, { uv: [ 0.3, 0.4 ] } ];

			await assignTeacherTargets( samples, teacher );

			expect( samples[ 0 ].target ).toEqual( [ 1, 2, 0 ] );
			expect( samples[ 1 ].target ).toEqual( [ 3, 4, 0 ] );

		} );

		it( 'slices a target array longer than 3 components down to exactly 3', async () => {

			const teacher = createBatchTeacher( () => [ 1, 2, 3, 4, 5 ] );
			const samples = [ { uv: [ 0, 0 ] } ];

			await assignTeacherTargets( samples, teacher );

			expect( samples[ 0 ].target ).toEqual( [ 1, 2, 3 ] );

		} );

		it( 'handles an empty sample list without throwing', async () => {

			const teacher = createBatchTeacher( () => [ 0, 0, 0 ] );
			const samples = [];

			await expect( assignTeacherTargets( samples, teacher ) ).resolves.toBeUndefined();
			expect( samples ).toEqual( [] );

		} );

	} );

	describe( 'normalizeDirectLightingTargets', () => {

		it( 'divides target by nDotL and sets weight = nDotL when nDotL >= minimumCosine (default 0.05)', () => {

			const samples = [ {
				wi: [ 0, 0, 0.5 ],
				target: [ 1, 2, 3 ]
			} ];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].directTarget ).toEqual( [ 1, 2, 3 ] );
			expect( samples[ 0 ].target ).toEqual( [ 2, 4, 6 ] );
			expect( samples[ 0 ].weight ).toBeCloseTo( 0.5, 10 );

		} );

		it( 'zeroes target and weight when nDotL is below the minimum cosine threshold', () => {

			const samples = [ {
				wi: [ 0, 0, 0.01 ], // below default 0.05 threshold
				target: [ 1, 2, 3 ]
			} ];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].target ).toEqual( [ 0, 0, 0 ] );
			expect( samples[ 0 ].weight ).toBe( 0 );
			// directTarget still preserves the original (valid) teacher target
			expect( samples[ 0 ].directTarget ).toEqual( [ 1, 2, 3 ] );

		} );

		it( 'a grazing-exact wi[2] equal to the minimum cosine is included (inclusive lower boundary)', () => {

			const samples = [ {
				wi: [ 0, 0, 0.05 ],
				target: [ 1, 1, 1 ]
			} ];

			normalizeDirectLightingTargets( samples, 0.05 );

			expect( samples[ 0 ].weight ).toBeCloseTo( 0.05, 10 );
			expect( samples[ 0 ].target ).toEqual( [ 20, 20, 20 ] );

		} );

		it( 'clamps a negative wi[2] (light below the surface) to nDotL = 0, zeroing target/weight', () => {

			const samples = [ {
				wi: [ 0, 0, -0.9 ],
				target: [ 5, 5, 5 ]
			} ];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].target ).toEqual( [ 0, 0, 0 ] );
			expect( samples[ 0 ].weight ).toBe( 0 );
			// the raw teacher target was still finite/valid, so directTarget survives
			expect( samples[ 0 ].directTarget ).toEqual( [ 5, 5, 5 ] );

		} );

		it( 'treats a non-finite target as invalid: directTarget becomes null and target/weight are zeroed regardless of nDotL', () => {

			const samples = [ {
				wi: [ 0, 0, 1 ], // nDotL = 1, well above threshold
				target: [ NaN, 2, 3 ]
			} ];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].directTarget ).toBeNull();
			expect( samples[ 0 ].target ).toEqual( [ 0, 0, 0 ] );
			expect( samples[ 0 ].weight ).toBe( 0 );

		} );

		it( 'treats an Infinity-valued target as invalid the same way as NaN', () => {

			const samples = [ {
				wi: [ 0, 0, 1 ],
				target: [ 1, Infinity, 3 ]
			} ];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].directTarget ).toBeNull();
			expect( samples[ 0 ].target ).toEqual( [ 0, 0, 0 ] );
			expect( samples[ 0 ].weight ).toBe( 0 );

		} );

		it( 'treats a target with fewer than 3 components as invalid', () => {

			const samples = [ {
				wi: [ 0, 0, 1 ],
				target: [ 1, 2 ]
			} ];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].directTarget ).toBeNull();
			expect( samples[ 0 ].target ).toEqual( [ 0, 0, 0 ] );
			expect( samples[ 0 ].weight ).toBe( 0 );

		} );

		it( 'a fully zero target at grazing-but-valid nDotL normalizes to a zero target with nonzero weight', () => {

			const samples = [ {
				wi: [ 0, 0, 0.5 ],
				target: [ 0, 0, 0 ]
			} ];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].directTarget ).toEqual( [ 0, 0, 0 ] );
			expect( samples[ 0 ].target ).toEqual( [ 0, 0, 0 ] ); // 0 / 0.5 = 0
			expect( samples[ 0 ].weight ).toBeCloseTo( 0.5, 10 );

		} );

		it( 'applies highlightLossScale to up-weight samples with high peak radiance, against an independently hand-computed literal', () => {

			// Hand-derived by hand outside of, and independent from, the source formula:
			// peak channel of the direct target is 9. With scale 2, the fraction
			// 9 parts-of-10 (9 out of a peak-plus-one denominator of 10) contributes
			// 2 * 0.9 = 1.8 of extra weight on top of the baseline 1, giving a
			// highlight multiplier of 2.8. That multiplier scales the geometric
			// nDotL term of 0.5, giving a final weight of 0.5 * 2.8 = 1.4.
			const samples = [ {
				wi: [ 0, 0, 0.5 ],
				target: [ 9, 1, 0 ]
			} ];

			normalizeDirectLightingTargets( samples, 0.05, 2 );

			expect( samples[ 0 ].weight ).toBeCloseTo( 1.4, 10 );

		} );

		it( 'a smaller, distinct highlightLossScale/peak-radiance combination also matches an independently hand-computed literal', () => {

			// Independent second data point (different scale and peak) so the two
			// highlightLossScale tests can't both pass merely by miscopying the
			// same source formula: peak channel is 3, scale is 1, so the fraction
			// is 3 parts-of-4 = 0.75, multiplier is 1 + 1*0.75 = 1.75, and with
			// nDotL of 0.2 the final weight is 0.2 * 1.75 = 0.35.
			const samples = [ {
				wi: [ 0, 0, 0.2 ],
				target: [ 0, 3, 2 ]
			} ];

			normalizeDirectLightingTargets( samples, 0.05, 1 );

			expect( samples[ 0 ].weight ).toBeCloseTo( 0.35, 10 );

		} );

		it( 'highlightLossScale = 0 (default) leaves weight equal to plain nDotL regardless of peak radiance', () => {

			const samples = [ {
				wi: [ 0, 0, 0.5 ],
				target: [ 1000, 1000, 1000 ] // extreme peak radiance
			} ];

			normalizeDirectLightingTargets( samples, 0.05, 0 );

			expect( samples[ 0 ].weight ).toBeCloseTo( 0.5, 10 );

		} );

		it( 'an extremely large peak radiance saturates the highlight weight toward 1 + highlightLossScale, never exceeding it', () => {

			const highlightLossScale = 3;
			const samples = [ {
				wi: [ 0, 0, 1 ],
				target: [ 1e12, 0, 0 ]
			} ];

			normalizeDirectLightingTargets( samples, 0.05, highlightLossScale );

			// peakRadiance / (1 + peakRadiance) -> 1 as peakRadiance -> infinity
			expect( samples[ 0 ].weight ).toBeLessThanOrEqual( 1 + highlightLossScale + 1e-9 );
			expect( samples[ 0 ].weight ).toBeGreaterThan( 1 + highlightLossScale - 1e-6 );

		} );

		it( 'a negative-valued target channel does not raise peakRadiance below the floor of 0', () => {

			// peakRadiance = max(-5, -2, -8, 0) = 0, so highlightWeight collapses to 1 (no boost)
			const samples = [ {
				wi: [ 0, 0, 0.5 ],
				target: [ -5, -2, -8 ]
			} ];

			normalizeDirectLightingTargets( samples, 0.05, 10 );

			expect( samples[ 0 ].weight ).toBeCloseTo( 0.5, 10 ); // nDotL * 1, no highlight boost

		} );

		it( 'processes multiple samples independently in a single call', () => {

			const samples = [
				{ wi: [ 0, 0, 0.5 ], target: [ 2, 2, 2 ] },
				{ wi: [ 0, 0, 0.01 ], target: [ 2, 2, 2 ] },
				{ wi: [ 0, 0, 1 ], target: [ NaN, 0, 0 ] }
			];

			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].weight ).toBeCloseTo( 0.5, 10 );
			expect( samples[ 1 ].weight ).toBe( 0 );
			expect( samples[ 2 ].weight ).toBe( 0 );
			expect( samples[ 2 ].directTarget ).toBeNull();

		} );

		it( 'a custom minimumCosine threshold is respected instead of the default', () => {

			const samples = [ { wi: [ 0, 0, 0.3 ], target: [ 1, 1, 1 ] } ];

			normalizeDirectLightingTargets( samples, 0.5 ); // 0.3 < 0.5 -> rejected

			expect( samples[ 0 ].weight ).toBe( 0 );
			expect( samples[ 0 ].target ).toEqual( [ 0, 0, 0 ] );

		} );

	} );

	describe( 'assignTeacherTargets + normalizeDirectLightingTargets combined (fake-teacher pipeline)', () => {

		it( 'a teacher-produced target correctly flows through normalization end to end', async () => {

			const teacher = createBatchTeacher( () => [ 0.4, 0.6, 0.8 ] );
			const samples = [ { uv: [ 0.5, 0.5 ], wi: [ 0, 0, 0.5 ] } ];

			await assignTeacherTargets( samples, teacher );
			normalizeDirectLightingTargets( samples );

			expect( samples[ 0 ].directTarget ).toEqual( [ 0.4, 0.6, 0.8 ] );
			expect( samples[ 0 ].target ).toEqual( [ 0.8, 1.2, 1.6 ] );
			expect( samples[ 0 ].weight ).toBeCloseTo( 0.5, 10 );

		} );

	} );

} );

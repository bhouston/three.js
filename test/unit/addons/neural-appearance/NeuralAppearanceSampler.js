import { describe, test, expect } from 'vitest';
import {
	generateTrainingSamples,
	generateIBLTrainingSamples,
	generateValidationSamples,
	normalizeDirectLightingTargets
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceSampler.js';

function createMockTeacher() {

	return {
		supportsEmission: false,
		supportsOpacity: false,
		encodeInputs( uv ) {

			return [ uv[ 0 ], uv[ 1 ] ];

		},
		async evaluateBatch( samples ) {

			return samples.map( ( sample ) => [
				Math.max( sample.uv[ 0 ] * sample.wi[ 2 ], 0 ),
				Math.max( sample.uv[ 1 ] * sample.wo[ 2 ], 0 ),
				Math.max( sample.wi[ 0 ] * 0.5 + 0.5, 0 )
			] );

		}
	};

}

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceSampler', () => {

			test( 'stabilizes direct-light targets near grazing incidence', () => {

				const samples = [
					{ wi: [ 0, 0, 0.5 ], target: [ 0.25, 0.5, 1 ] },
					{ wi: [ 1, 0, 0.01 ], target: [ 1, 2, 3 ] },
					{ wi: [ 0, 0, 1 ], target: [ 1, NaN, 3 ] }
				];

				normalizeDirectLightingTargets( samples, 0.05 );

				expect( samples[ 0 ].target ).toEqual( [ 0.5, 1, 2 ] );
				expect( samples[ 0 ].weight ).toBe( 0.5 );
				expect( samples[ 1 ].target ).toEqual( [ 0, 0, 0 ] );
				expect( samples[ 1 ].weight ).toBe( 0 );
				expect( samples[ 2 ].weight ).toBe( 0 );

				const highlightSamples = [
					{ wi: [ 0, 0, 0.5 ], target: [ 0.25, 0.5, 1 ] }
				];
				normalizeDirectLightingTargets( highlightSamples, 0.05, 2 );
				expect( highlightSamples[ 0 ].weight > 0.5 && highlightSamples[ 0 ].weight < 1.5 ).toBeTruthy();

			} );

			test( 'oversamples valid specular direction pairs', async () => {

				let state = 19;
				const random = () => {

					state = ( state * 1664525 + 1013904223 ) >>> 0;
					return state / 4294967296;

				};

				const samples = await generateTrainingSamples( {
					batchSize: 512,
					colorAugmentation: false,
					minimumTrainingCosine: 0.05,
					highlightLossScale: 2
				}, createMockTeacher(), random );
				const mirrorPairs = samples.filter( ( sample ) =>
					Math.abs( sample.wi[ 0 ] + sample.wo[ 0 ] ) < 1e-10 &&
					Math.abs( sample.wi[ 1 ] + sample.wo[ 1 ] ) < 1e-10 &&
					Math.abs( sample.wi[ 2 ] - sample.wo[ 2 ] ) < 1e-10
				);

				expect( samples.length ).toBe( 512 );
				expect( mirrorPairs.length >= 50 ).toBeTruthy();
				expect( samples.every( ( sample ) => sample.wo[ 2 ] >= 0 ) ).toBeTruthy();

			} );

			test( 'generates deterministic validation samples spanning grazing angles', async () => {

				const teacher = createMockTeacher();
				const samples = await generateValidationSamples( {
					batchSize: 16,
					minimumTrainingCosine: 0.05
				}, teacher );

				const wiCosines = [ ...new Set( samples.map( ( s ) => s.wi[ 2 ] ) ) ].sort();
				expect( wiCosines ).toEqual( [ 0.025, 0.1, 0.4, 0.8 ] );
				expect( samples.every( ( s ) => s.target !== undefined ) ).toBeTruthy();
				expect( samples.every( ( s ) => s.iblWeight === 0 ) ).toBeTruthy();
				expect( samples.every( ( s ) => Array.isArray( s.iblDirection ) && s.iblDirection.length === 3 ) ).toBeTruthy();
				expect( samples.every( ( s ) => Number.isFinite( s.iblRoughness ) ) ).toBeTruthy();

			} );

			test( 'generates grouped quadrature samples for IBL refinement', async () => {

				let state = 31;
				const random = () => {

					state = ( state * 1664525 + 1013904223 ) >>> 0;
					return state / 4294967296;

				};

				const samples = await generateIBLTrainingSamples( {
					batchSize: 16,
					iblSampleCount: 4,
					iblIntegrationSamples: 8
				}, createMockTeacher(), random );

				expect( samples.length ).toBe( 4 );
				expect( samples.every( ( sample ) => sample.iblWeight === 0 ) ).toBeTruthy();
				expect( samples.every( ( sample ) => Array.isArray( sample.iblIncoming ) && sample.iblIncoming.length === 3 ) ).toBeTruthy();
				expect( samples.every( ( sample ) => Array.isArray( sample.iblIrradiance ) && sample.iblIrradiance.length === 3 ) ).toBeTruthy();
				expect( samples.every( ( sample ) => Array.isArray( sample.iblIndirectRadiance ) && sample.iblIndirectRadiance.length === 3 ) ).toBeTruthy();
				expect( samples.every( ( sample ) => Array.isArray( sample.iblIndirectIrradiance ) && sample.iblIndirectIrradiance.length === 3 ) ).toBeTruthy();

			} );

			test( 'assigns teacher IBL query, incoming, irradiance, and isolate indirect labels', async () => {

				const teacher = {
					supportsIBL: true,
					encodeInputs( uv ) {

						return [ uv[ 0 ], uv[ 1 ] ];

					},
					async evaluateBatch( samples, mode ) {

						if ( mode === 'iblQuery' ) return samples.map( ( sample ) => [ sample.normal[ 0 ], sample.normal[ 1 ], sample.normal[ 2 ], 0.25 ] );
						if ( mode === 'iblIncoming' ) return samples.map( () => [ 0.1, 0.2, 0.3 ] );
						if ( mode === 'iblIrradiance' ) return samples.map( () => [ 0.7, 0.8, 0.9 ] );
						if ( mode === 'iblIndirectRadiance' ) return samples.map( () => [ 0.4, 0.5, 0.6 ] );
						if ( mode === 'iblIndirectIrradiance' ) return samples.map( () => [ 0.15, 0.05, 0.02 ] );

						return samples.map( () => [ 0.2, 0.3, 0.4 ] );

					}
				};
				const samples = await generateValidationSamples( {
					batchSize: 4,
					minimumTrainingCosine: 0.05
				}, teacher );

				expect( samples.every( ( sample ) => sample.iblWeight === 1 ) ).toBeTruthy();
				expect( Math.abs( samples[ 0 ].iblDirection[ 0 ] ) < 1e-6 ).toBeTruthy();
				expect( Math.abs( samples[ 0 ].iblDirection[ 1 ] ) < 1e-6 ).toBeTruthy();
				expect( Math.abs( samples[ 0 ].iblDirection[ 2 ] - 1 ) < 1e-6 ).toBeTruthy();
				expect( samples[ 0 ].iblRoughness ).toBe( 0.25 );
				expect( samples[ 0 ].iblIncoming ).toEqual( [ 0.1, 0.2, 0.3 ] );
				expect( samples[ 0 ].iblIrradiance ).toEqual( [ 0.7, 0.8, 0.9 ] );
				expect( samples[ 0 ].iblIndirectRadiance ).toEqual( [ 0.4, 0.5, 0.6 ] );
				expect( samples[ 0 ].iblIndirectIrradiance ).toEqual( [ 0.15, 0.05, 0.02 ] );

			} );

		} );

	} );

} );

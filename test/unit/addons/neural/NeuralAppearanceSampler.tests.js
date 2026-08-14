import {
	generateTrainingSamples,
	generateValidationSamples,
	normalizeDirectLightingTargets,
	fitIBLTargetFromRecords
} from '../../../../examples/jsm/neural/NeuralAppearanceSampler.js';

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

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceSampler', () => {

			QUnit.test( 'stabilizes direct-light targets near grazing incidence', ( assert ) => {

				const samples = [
					{ wi: [ 0, 0, 0.5 ], target: [ 0.25, 0.5, 1 ] },
					{ wi: [ 1, 0, 0.01 ], target: [ 1, 2, 3 ] },
					{ wi: [ 0, 0, 1 ], target: [ 1, NaN, 3 ] }
				];

				normalizeDirectLightingTargets( samples, 0.05 );

				assert.deepEqual( samples[ 0 ].target, [ 0.5, 1, 2 ], 'normalizes a well-conditioned target' );
				assert.strictEqual( samples[ 0 ].weight, 0.5, 'weights the loss by the conditioning cosine' );
				assert.deepEqual( samples[ 1 ].target, [ 0, 0, 0 ], 'discards an ill-conditioned grazing target' );
				assert.strictEqual( samples[ 1 ].weight, 0, 'gives the grazing target no optimizer influence' );
				assert.strictEqual( samples[ 2 ].weight, 0, 'rejects a non-finite teacher target' );

				const highlightSamples = [
					{ wi: [ 0, 0, 0.5 ], target: [ 0.25, 0.5, 1 ] }
				];
				normalizeDirectLightingTargets( highlightSamples, 0.05, 2 );
				assert.ok( highlightSamples[ 0 ].weight > 0.5 && highlightSamples[ 0 ].weight < 1.5, 'bounded highlight weighting emphasizes bright, well-conditioned targets' );

			} );

			QUnit.test( 'oversamples valid specular direction pairs', async ( assert ) => {

				let state = 19;
				const random = () => {

					state = ( state * 1664525 + 1013904223 ) >>> 0;
					return state / 4294967296;

				};

				const samples = await generateTrainingSamples( {
					resolution: 1,
					iterations: 1,
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

				assert.ok( mirrorPairs.length >= 50, 'dedicates a substantial part of each batch to the sharp specular ridge' );
				assert.ok( samples.every( ( sample ) => sample.wo[ 2 ] >= 0 ), 'keeps outgoing directions in the visible hemisphere' );

			} );

			QUnit.test( 'samples mip levels exponentially in favor of fine levels', async ( assert ) => {

				let state = 23;
				const random = () => {

					state = ( state * 1664525 + 1013904223 ) >>> 0;
					return state / 4294967296;

				};

				const samples = await generateTrainingSamples( {
					resolution: 8,
					iterations: 1,
					batchSize: 1024,
					fixedTrainingMip: - 1,
					mipSamplingDecay: 0.5,
					colorAugmentation: false,
					minimumTrainingCosine: 0.05,
					highlightLossScale: 2
				}, createMockTeacher(), random );
				const counts = [ 0, 0, 0, 0 ];

				for ( const sample of samples ) counts[ sample.mip ] ++;

				assert.strictEqual( samples.length, 4096, 'preserves the prior total record budget while choosing one mip per record' );
				assert.ok( counts[ 0 ] > counts[ 1 ] && counts[ 1 ] > counts[ 2 ] && counts[ 2 ] > counts[ 3 ], 'favors each finer level over the next coarser level' );

			} );

			QUnit.test( 'generates deterministic validation samples spanning grazing angles', async ( assert ) => {

				const teacher = createMockTeacher();
				const samples = await generateValidationSamples( {
					resolution: 2,
					batchSize: 16,
					fixedTrainingMip: - 1,
					minimumTrainingCosine: 0.05
				}, teacher );

				const wiCosines = [ ...new Set( samples.map( ( s ) => s.wi[ 2 ] ) ) ].sort();
				assert.deepEqual( wiCosines, [ 0.025, 0.1, 0.4, 0.8 ], 'sweeps through defined grazing cosines' );
				assert.ok( samples.every( ( s ) => s.target !== undefined ), 'attaches evaluated teacher targets' );
				assert.ok( samples.every( ( s ) => Array.isArray( s.iblTarget ) && s.iblTarget.length === 13 ), 'attaches local-frame IBL targets' );

			} );

			QUnit.test( 'fits IBL specular direction to the local reflection of wo', ( assert ) => {

				const wo = [ 0.6, 0, 0.8 ];
				const records = [
					{ wi: [ 0, 0, 1 ], wo, target: [ 0.1, 0.1, 0.1 ] },
					{ wi: [ - 0.6, 0, 0.8 ], wo, target: [ 2, 2, 2 ] },
					{ wi: [ 0.5, 0.5, 0.707 ], wo, target: [ 0.2, 0.2, 0.2 ] }
				];
				const target = fitIBLTargetFromRecords( records );

				assert.deepEqual( target.slice( 0, 3 ), [ 0, 0, 1 ], 'diffuse direction is the canonical normal' );
				assert.ok( Math.abs( target[ 6 ] + 0.6 ) < 1e-6, 'specular x is the reflection of wo' );
				assert.ok( Math.abs( target[ 7 ] ) < 1e-6, 'specular y is the reflection of wo' );
				assert.ok( Math.abs( target[ 8 ] - 0.8 ) < 1e-6, 'specular z is the reflection of wo' );
				assert.ok( target[ 3 ] + target[ 10 ] > 0, 'splits directional albedo across diffuse and specular weights' );

			} );

		} );

	} );

} );

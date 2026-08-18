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
				assert.ok( samples.every( ( s ) => s.iblWeight === 0 ), 'disables IBL labels when the teacher has no environment' );
				assert.ok( samples.every( ( s ) => Array.isArray( s.iblDirection ) && s.iblDirection.length === 3 ), 'attaches IBL query directions' );
				assert.ok( samples.every( ( s ) => Number.isFinite( s.iblRoughness ) ), 'attaches IBL query roughness' );

			} );

			QUnit.test( 'generates grouped quadrature samples for IBL refinement', async ( assert ) => {

				let state = 31;
				const random = () => {

					state = ( state * 1664525 + 1013904223 ) >>> 0;
					return state / 4294967296;

				};

				const samples = await generateIBLTrainingSamples( {
					resolution: 4,
					batchSize: 16,
					iblSampleCount: 4,
					iblIntegrationSamples: 8
				}, createMockTeacher(), random );

				assert.strictEqual( samples.length, 4, 'returns one trainable query per IBL sample' );
				assert.ok( samples.every( ( sample ) => sample.iblWeight === 0 ), 'disables IBL labels when the teacher has no environment' );
				assert.ok( samples.every( ( sample ) => Array.isArray( sample.iblIncoming ) && sample.iblIncoming.length === 3 ), 'attaches incoming IBL radiance slots' );
				assert.ok( samples.every( ( sample ) => Array.isArray( sample.iblIrradiance ) && sample.iblIrradiance.length === 3 ), 'attaches incoming IBL irradiance slots' );
				assert.ok( samples.every( ( sample ) => Array.isArray( sample.iblIndirectRadiance ) && sample.iblIndirectRadiance.length === 3 ), 'attaches radiance IBL slots' );
				assert.ok( samples.every( ( sample ) => Array.isArray( sample.iblIndirectIrradiance ) && sample.iblIndirectIrradiance.length === 3 ), 'attaches irradiance IBL slots' );

			} );

			QUnit.test( 'assigns teacher IBL query, incoming, irradiance, and isolate indirect labels', async ( assert ) => {

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
					resolution: 1,
					batchSize: 4,
					fixedTrainingMip: 0,
					minimumTrainingCosine: 0.05
				}, teacher );

				assert.ok( samples.every( ( sample ) => sample.iblWeight === 1 ), 'enables IBL loss when the teacher can sample the environment' );
				assert.ok( Math.abs( samples[ 0 ].iblDirection[ 0 ] ) < 1e-6, 'canonicalizes IBL query x' );
				assert.ok( Math.abs( samples[ 0 ].iblDirection[ 1 ] ) < 1e-6, 'canonicalizes IBL query y' );
				assert.ok( Math.abs( samples[ 0 ].iblDirection[ 2 ] - 1 ) < 1e-6, 'canonicalizes IBL query z to the shading normal' );
				assert.strictEqual( samples[ 0 ].iblRoughness, 0.25, 'stores the teacher IBL roughness used for PMREM' );
				assert.deepEqual( samples[ 0 ].iblIncoming, [ 0.1, 0.2, 0.3 ], 'stores incoming environment radiance' );
				assert.deepEqual( samples[ 0 ].iblIrradiance, [ 0.7, 0.8, 0.9 ], 'stores incoming environment irradiance' );
				assert.deepEqual( samples[ 0 ].iblIndirectRadiance, [ 0.4, 0.5, 0.6 ], 'stores teacher radiance-only IBL' );
				assert.deepEqual( samples[ 0 ].iblIndirectIrradiance, [ 0.15, 0.05, 0.02 ], 'stores teacher irradiance-only IBL' );

			} );

		} );

	} );

} );

import { DataUtils } from 'three';
import {
	NeuralAppearanceTrainer,
	evaluateNeuralAppearanceJson,
	generateTrainingSamples,
	normalizeDirectLightingTargets
} from '../../../../examples/jsm/materials/NeuralAppearanceTrainer.js';
import { NeuralAppearanceTeacherEvaluator } from '../../../../examples/jsm/materials/NeuralAppearanceTeacherEvaluator.js';

function createBatchTeacher() {

	const calls = [];

	return {
		calls,
		async init() {},
		encodeInputs( uv = [ 0.5, 0.5 ] ) {

			return [ uv[ 0 ], uv[ 1 ] ];

		},
		async evaluateBatch( samples ) {

			calls.push( samples.map( ( sample ) => ( {
				uv: sample.uv.slice(),
				wi: sample.wi.slice(),
				wo: sample.wo.slice(),
				normal: sample.normal.slice(),
				tangent: sample.tangent.slice(),
				bitangent: sample.bitangent.slice(),
				duvDx: sample.duvDx.slice(),
				duvDy: sample.duvDy.slice(),
				mip: sample.mip
			} ) ) );

			return samples.map( ( sample ) => [
				Math.max( sample.uv[ 0 ] * sample.wi[ 2 ], 0 ),
				Math.max( sample.uv[ 1 ] * sample.wo[ 2 ], 0 ),
				Math.max( sample.wi[ 0 ] * 0.5 + 0.5, 0 )
			] );

		}
	};

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Materials', () => {

		QUnit.module( 'NeuralAppearanceTrainer', () => {

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
				}, createBatchTeacher(), random );
				const mirrorPairs = samples.filter( ( sample ) =>
					Math.abs( sample.wi[ 0 ] + sample.wo[ 0 ] ) < 1e-10 &&
					Math.abs( sample.wi[ 1 ] + sample.wo[ 1 ] ) < 1e-10 &&
					Math.abs( sample.wi[ 2 ] - sample.wo[ 2 ] ) < 1e-10
				);

				assert.ok( mirrorPairs.length >= 50, 'dedicates a substantial part of each batch to the sharp specular ridge' );
				assert.ok( samples.every( ( sample ) => sample.wo[ 2 ] >= 0 ), 'keeps outgoing directions in the visible hemisphere' );

			} );

			QUnit.test( 'requests asynchronous batched teacher targets', async ( assert ) => {

				const teacher = createBatchTeacher();
				const trainer = new NeuralAppearanceTrainer( {
					resolution: 2,
					iterations: 1,
					batchSize: 4,
					hiddenSize: 4,
					yieldEvery: 0,
					seed: 7
				} );

				let progressJson = null;
				const result = await trainer.train( {
					material: {},
					teacher,
					onProgress: ( progress ) => {

						progressJson = progress.json;

					}
				} );

				assert.ok( teacher.calls.length >= 3, 'teacher used for validation, training, and exported references' );
				assert.strictEqual( teacher.calls[ 0 ][ 0 ].normal[ 2 ], 1, 'canonical normal passed to teacher' );
				assert.strictEqual( teacher.calls[ 0 ][ 0 ].tangent[ 0 ], 1, 'canonical tangent passed to teacher' );
				assert.strictEqual( teacher.calls[ 0 ][ 0 ].bitangent[ 1 ], 1, 'canonical bitangent passed to teacher' );
				assert.strictEqual( progressJson.format, 'three-neural-appearance', 'reports a renderable model snapshot during training' );
				assert.strictEqual( progressJson.latents.textures.length, 2, 'includes both live latent textures' );
				assert.strictEqual( result.json.referenceEvaluations.length, 3, 'exports reference samples' );
				assert.strictEqual( result.json.referenceEvaluations[ 0 ].targetRgb.length, 3, 'keeps GPU teacher target with exported reference' );
				assert.ok( result.model.rotationWeights.some( ( value ) => value !== 0 ), 'trains learned-frame weights' );
				assert.ok( result.model.rotationM.some( ( value ) => value !== 0 ), 'tracks learned-frame Adam moments' );
				assert.deepEqual( result.json.decoder.rotation.weights, result.model.rotationWeights, 'exports trained learned-frame weights' );
				assert.strictEqual( result.validation.loss, result.validationLoss, 'reports validation from the serialized runtime evaluator' );
				assert.strictEqual( result.validation.mipLevels, 2, 'validates every exported mip level' );
				assert.strictEqual( result.validation.sampleCount, 8, 'validates the held-out samples across both mips' );
				assert.strictEqual( result.validation.preview.samples.length, 8, 'reports preview samples for the held-out teacher inputs' );
				assert.strictEqual( result.validation.preview.samples[ 0 ].targetRgb.length, 3, 'keeps teacher RGB in the validation preview' );
				assert.strictEqual( result.validation.preview.samples[ 0 ].predictionRgb.length, 3, 'keeps neural RGB in the validation preview' );
				assert.strictEqual( result.validation.preview.samples[ 0 ].wi.length, 3, 'keeps the preview light direction' );
				assert.strictEqual( result.validation.preview.samples[ 0 ].wo.length, 3, 'keeps the preview view direction' );
				assert.strictEqual( result.validation.angularBins.wi.length, 3, 'reports direct HDR error by incoming-angle bin' );
				assert.strictEqual( result.validation.angularBins.wo.length, 3, 'reports direct HDR error by outgoing-angle bin' );
				assert.strictEqual( result.validation.reciprocity.sampleCount, result.validation.sampleCount, 'checks every held-out prediction with exchanged light and view directions' );
				assert.strictEqual( result.validation.angularSmoothness.sampleCount, result.validation.sampleCount * 2, 'checks local incoming- and outgoing-direction perturbations' );
				assert.ok( Number.isFinite( result.validation.reciprocity.meanAbsoluteDifference ), 'reports a finite reciprocity error' );
				assert.ok( Number.isFinite( result.validation.angularSmoothness.meanAbsoluteDifference ), 'reports a finite angular smoothness error' );

			} );

			QUnit.test( 'supports a one-texel baseline and deterministic grazing validation', async ( assert ) => {

				const teacher = createBatchTeacher();
				const trainer = new NeuralAppearanceTrainer( {
					resolution: 1,
					iterations: 1,
					batchSize: 16,
					hiddenSize: 4,
					yieldEvery: 0,
					seed: 5
				} );
				const result = await trainer.train( { material: {}, teacher } );
				const validationCall = teacher.calls[ 1 ];
				const wiCosines = [ ...new Set( validationCall.map( ( sample ) => sample.wi[ 2 ] ) ) ].sort();
				const woCosines = [ ...new Set( validationCall.map( ( sample ) => sample.wo[ 2 ] ) ) ].sort();

				assert.deepEqual( result.model.latentGrids.map( ( grid ) => [ grid.width, grid.height ] ), [[ 1, 1 ]], 'trains one latent texel and mip zero when requested' );
				assert.deepEqual( wiCosines, [ 0.025, 0.1, 0.4, 0.8 ], 'validation sweeps incoming directions through grazing angles' );
				assert.deepEqual( woCosines, [ 0.025, 0.1, 0.4, 0.8 ], 'validation sweeps outgoing directions through grazing angles' );
				assert.strictEqual( result.validation.preview.samples.length, 16, 'keeps the original held-out samples in the visual preview' );
				assert.strictEqual( result.validation.directional.preview.samples.length, 0, 'reports the directional grid separately from the visual preview' );

			} );

			QUnit.test( 'requires training-safe output and gradient settings', async ( assert ) => {

				const teacher = createBatchTeacher();

				await assert.rejects(
					new NeuralAppearanceTrainer( { outputActivation: { type: 'exp' } } ).train( { material: {}, teacher } ),
					/Only linear output activation/,
					'rejects an output activation whose derivative is not trained'
				);
				await assert.rejects(
					new NeuralAppearanceTrainer( { maxGradientNorm: Infinity } ).train( { material: {}, teacher } ),
					/maxGradientNorm must be finite/,
					'requires active finite gradient clipping'
				);

			} );

			QUnit.test( 'rejects non-half-float teacher readback', async ( assert ) => {

				const renderer = {
					toneMapping: 0,
					getRenderTarget: () => null,
					getClearAlpha: () => 1,
					getClearColor() {},
					setClearColor() {},
					setRenderTarget() {},
					render() {},
					readRenderTargetPixelsAsync: async () => new Uint8Array( 4 )
				};
				const evaluator = new NeuralAppearanceTeacherEvaluator( {}, renderer );

				evaluator._target = {};
				evaluator._atlasWidth = 1;
				evaluator._atlasHeight = 1;

				await assert.rejects(
					evaluator._renderAndRead(),
					/Half-float teacher readback is required/,
					'does not silently train from LDR pixels'
				);

			} );

			QUnit.test( 'evaluates exported half-float latents with runtime LOD selection', ( assert ) => {

				const outputWeights = new Array( 3 * 20 ).fill( 0 );
				outputWeights[ 0 ] = 1;
				outputWeights[ 20 + 1 ] = 1;
				outputWeights[ 40 + 2 ] = 1;
				const json = {
					latents: {
						textures: [
							{
								wrap: 'repeat',
								mipmaps: [
									{ width: 2, height: 2, data: [ 0.1, 0.2, 0.3, 0, 0.4, 0.5, 0.6, 0, 0.7, 0.8, 0.9, 0, 1, 1.1, 1.2, 0 ] },
									{ width: 1, height: 1, data: [ 0.75, 0.5, 0.25, 0 ] }
								]
							},
							{
								wrap: 'repeat',
								mipmaps: [
									{ width: 2, height: 2, data: new Array( 16 ).fill( 0 ) },
									{ width: 1, height: 1, data: [ 0, 0, 0, 0 ] }
								]
							}
						]
					},
					decoder: {
						rotation: { weights: new Array( 8 * 12 ).fill( 0 ) },
						layers: [ {
							inputSize: 20,
							outputSize: 3,
							activation: 'linear',
							weights: outputWeights,
							biases: [ 0, 0, 0 ]
						} ],
						outputActivation: { type: 'linear' }
					}
				};
				const direction = [ 0, 0, 1 ];
				const base = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: direction,
					wo: direction,
					duvDx: [ 0, 0 ],
					duvDy: [ 0, 0 ]
				} );
				const coarse = evaluateNeuralAppearanceJson( json, {
					uv: [ 0.25, 0.25 ],
					wi: direction,
					wo: direction,
					duvDx: [ 1, 0 ],
					duvDy: [ 0, 1 ]
				} );
				const half = ( value ) => DataUtils.fromHalfFloat( DataUtils.toHalfFloat( value ) );

				assert.deepEqual( base, [ half( 0.1 ), half( 0.2 ), half( 0.3 ) ], 'samples the base mip at the requested UV after half-float conversion' );
				assert.deepEqual( coarse, [ half( 0.75 ), half( 0.5 ), half( 0.25 ) ], 'selects the coarse mip from the runtime UV footprint' );

			} );

			QUnit.test( 'trains and exports appearance-filtered latent mip levels', async ( assert ) => {

				const teacher = createBatchTeacher();
				const trainer = new NeuralAppearanceTrainer( {
					resolution: 4,
					iterations: 1,
					batchSize: 6,
					hiddenSize: 4,
					yieldEvery: 0,
					seed: 11
				} );

				const result = await trainer.train( { material: {}, teacher } );
				const trainingCall = teacher.calls[ 2 ];
				const footprints = [ ...new Set( trainingCall.map( ( sample ) => sample.duvDx[ 0 ] ) ) ].sort();

				assert.deepEqual( footprints, [ 0.25, 0.5, 1 ], 'requests a matching teacher footprint for every mip' );
				assert.deepEqual( result.model.latentGrids.map( ( grid ) => [ grid.width, grid.height ] ), [[ 4, 4 ], [ 2, 2 ], [ 1, 1 ]], 'creates an independently optimized grid for every mip' );
				assert.ok( result.model.latentGrids.every( ( grid ) => grid.m.some( ( value ) => value !== 0 ) ), 'updates every mip during training' );
				assert.deepEqual( result.json.latents.textures[ 0 ].mipmaps.map( ( mip ) => [ mip.width, mip.height ] ), [[ 4, 4 ], [ 2, 2 ], [ 1, 1 ]], 'exports the trained mip hierarchy' );

				for ( let mip = 0; mip < result.model.latentGrids.length; mip ++ ) {

					const grid = result.model.latentGrids[ mip ];
					assert.deepEqual( result.json.latents.textures[ 0 ].mipmaps[ mip ].data.slice( 0, 4 ), grid.data.slice( 0, 4 ), `exports trained low channels for mip ${mip}` );
					assert.deepEqual( result.json.latents.textures[ 1 ].mipmaps[ mip ].data.slice( 0, 4 ), grid.data.slice( 4, 8 ), `exports trained high channels for mip ${mip}` );

				}

			} );

		} );

	} );

} );

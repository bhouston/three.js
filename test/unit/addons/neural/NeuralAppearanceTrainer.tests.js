import {
	NeuralAppearanceTrainer,
	estimateTrainingMemory,
	getTrainingSampleCapacity
} from '../../../../examples/jsm/neural/NeuralAppearanceTrainer.js';
import {
	createGpuMaterialTeacher,
	NeuralAppearanceTeacherEvaluator
} from '../../../../examples/jsm/neural/NeuralAppearanceTeacherEvaluator.js';

function createBatchTeacher( options = {} ) {

	const calls = [];

	return {
		calls,
		supportsEmission: options.supportsEmission === true,
		supportsOpacity: options.supportsOpacity === true,
		alphaCutoff: 0.5,
		async init() {},
		encodeInputs( uv = [ 0.5, 0.5 ] ) {

			return [ uv[ 0 ], uv[ 1 ] ];

		},
		async evaluateBatch( samples, targetMode = 'brdf' ) {

			calls.push( samples.map( ( sample ) => ( {
				targetMode,
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

			if ( targetMode === 'emission' ) {

				return samples.map( ( sample ) => [ sample.uv[ 0 ], sample.uv[ 1 ], 0.25 ] );

			}

			if ( targetMode === 'opacity' ) {

				return samples.map( ( sample ) => [ sample.uv[ 0 ] > 0.5 ? 1 : 0, 0, 0 ] );

			}

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

		QUnit.module( 'NeuralAppearanceTrainer', () => {

			QUnit.test( 'estimates scaled latent memory and fixed-mip training', ( assert ) => {

				const memory = estimateTrainingMemory( 64 );

				assert.strictEqual( memory.mipLevels, 7, 'counts the complete 64px mip hierarchy' );
				assert.strictEqual( memory.latentTexels, 5461, 'counts texels across independent mip grids' );
				assert.strictEqual( memory.trainingBytes, 5461 * 8 * 4 * 4, 'includes values, gradients, and two Adam moments' );
				assert.strictEqual( memory.exportBytes, 5461 * 8 * 2, 'estimates eight FP16 latent channels' );

			} );

			QUnit.test( 'sizes GPU training batches for sampled mip records', ( assert ) => {

				assert.strictEqual(
					getTrainingSampleCapacity( { resolution: 4, batchSize: 6, fixedTrainingMip: - 1 } ),
					18,
					'allocates one dispatch slot for each sampled mip training record'
				);
				assert.strictEqual(
					getTrainingSampleCapacity( { resolution: 4, batchSize: 6, fixedTrainingMip: 1 } ),
					6,
					'fixed-mip training only needs the requested batch size'
				);
				assert.strictEqual(
					getTrainingSampleCapacity( { resolution: 4, batchSize: 6, fixedTrainingMip: - 1, sampleAllMips: true } ),
					18,
					'allocates one dispatch slot for each explicit mip sample when sampleAllMips is enabled'
				);

			} );

			QUnit.test( 'requests asynchronous batched teacher targets and reports progress', async ( assert ) => {

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
				assert.deepEqual( result.json.outputs.brdf.rotation.weights, result.model.rotationWeights, 'exports trained learned-frame weights' );
				assert.strictEqual( result.validation.loss, result.validationLoss, 'reports validation from the serialized runtime evaluator' );
				assert.strictEqual( result.validation.mipLevels, 2, 'validates every exported mip level' );
				assert.strictEqual( result.validation.sampleCount, 8, 'validates the held-out samples across both mips' );

			} );

			QUnit.test( 'exports optional emission and opacity heads', async ( assert ) => {

				const teacher = createBatchTeacher( { supportsEmission: true, supportsOpacity: true } );
				const trainer = new NeuralAppearanceTrainer( {
					resolution: 2,
					iterations: 1,
					batchSize: 4,
					hiddenSize: 4,
					yieldEvery: 0,
					seed: 13
				} );
				const result = await trainer.train( { material: {}, teacher } );

				assert.ok( result.json.outputs.brdf, 'exports the BRDF head' );
				assert.ok( result.json.outputs.emission, 'exports the emission head' );
				assert.ok( result.json.outputs.opacity, 'exports the opacity head' );
				assert.strictEqual( result.json.outputs.opacity.alphaCutoff, 0.5, 'exports opacity cutoff metadata' );
				assert.ok( result.json.referenceEvaluations.some( ( reference ) => reference.targetEmission ), 'exports emission reference values' );
				assert.ok( result.json.referenceEvaluations.some( ( reference ) => Number.isFinite( reference.targetOpacity ) ), 'exports opacity reference values' );
				assert.ok( Number.isFinite( result.validation.emissionLoss ), 'reports emission validation loss' );
				assert.ok( Number.isFinite( result.validation.opacityLoss ), 'reports opacity validation loss' );

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

			QUnit.test( 'rejects unsupported GPU teacher materials', ( assert ) => {

				assert.throws(
					() => createGpuMaterialTeacher( { type: 'MeshBasicNodeMaterial' }, null ),
					/supported MeshPhysicalNodeMaterial teacher is required/,
					'does not distill Physical defaults for an unsupported surface material'
				);
				assert.ok(
					createGpuMaterialTeacher( { isMeshPhysicalNodeMaterial: true }, null ) instanceof NeuralAppearanceTeacherEvaluator,
					'accepts a physical node material teacher'
				);

			} );

			QUnit.test( 'trains sampled independent appearance mip levels', async ( assert ) => {

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
				const sampledMips = [ ...new Set( trainingCall.map( ( sample ) => sample.mip ) ) ];
				const updatedMips = result.model.latentGrids
					.map( ( grid, mip ) => grid.m.some( ( value ) => value !== 0 ) ? mip : - 1 )
					.filter( ( mip ) => mip >= 0 );

				assert.strictEqual( trainingCall.length, 18, 'preserves the all-mip record budget while sampling one mip per record' );
				assert.ok( footprints.every( ( footprint ) => [ 0.25, 0.5, 1 ].includes( footprint ) ), 'requests the footprint matching each sampled mip' );
				assert.deepEqual( result.model.latentGrids.map( ( grid ) => [ grid.width, grid.height ] ), [[ 4, 4 ], [ 2, 2 ], [ 1, 1 ]], 'creates an independently optimized grid for every mip' );
				assert.ok( updatedMips.length > 0 && updatedMips.every( ( mip ) => sampledMips.includes( mip ) ), 'only updates independently selected mip grids' );
				assert.deepEqual( result.json.latents.textures[ 0 ].mipmaps.map( ( mip ) => [ mip.width, mip.height ] ), [[ 4, 4 ], [ 2, 2 ], [ 1, 1 ]], 'exports the trained mip hierarchy' );

				for ( let mip = 0; mip < result.model.latentGrids.length; mip ++ ) {

					const grid = result.model.latentGrids[ mip ];
					assert.deepEqual( result.json.latents.textures[ 0 ].mipmaps[ mip ].data.slice( 0, 4 ), grid.data.slice( 0, 4 ), `exports trained low channels for mip ${mip}` );
					assert.deepEqual( result.json.latents.textures[ 1 ].mipmaps[ mip ].data.slice( 0, 4 ), grid.data.slice( 4, 8 ), `exports trained high channels for mip ${mip}` );

				}

			} );

			QUnit.test( 'supports backend configuration and validates GPU requirements', async ( assert ) => {

				const teacher = createBatchTeacher();

				// Explicit CPU backend
				const cpuTrainer = new NeuralAppearanceTrainer( {
					backend: 'cpu',
					resolution: 2,
					iterations: 1,
					batchSize: 4,
					hiddenSize: 4,
					yieldEvery: 0,
					seed: 1
				} );

				const cpuResult = await cpuTrainer.train( { material: {}, teacher } );
				assert.strictEqual( cpuResult.json.format, 'three-neural-appearance', 'trains successfully with explicit CPU backend' );

				// GPU backend without WebGPU renderer throws
				const gpuTrainer = new NeuralAppearanceTrainer( {
					backend: 'gpu',
					resolution: 2,
					iterations: 1,
					batchSize: 4,
					hiddenSize: 4
				} );

				await assert.rejects(
					gpuTrainer.train( { material: {}, teacher, renderer: null } ),
					/WebGPU renderer is required for GPU backend training/,
					'rejects GPU backend without WebGPU renderer'
				);

			} );

		} );

	} );

} );

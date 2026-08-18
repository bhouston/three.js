import {
	NeuralAppearanceTrainer,
	estimateTrainingMemory,
	getTrainingSampleCapacity
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTrainer.js';
import {
	createGpuMaterialTeacher,
	NeuralAppearanceTeacherEvaluator
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js';

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

			QUnit.test( 'requires training-safe output and gradient settings', async ( assert ) => {

				await assert.rejects(
					new NeuralAppearanceTrainer( { outputActivation: { type: 'exp' } } ).train( { material: {} } ),
					/Only linear output activation/,
					'rejects an output activation whose derivative is not trained'
				);
				await assert.rejects(
					new NeuralAppearanceTrainer( { maxGradientNorm: Infinity } ).train( { material: {} } ),
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

			QUnit.test( 'requires WebGPU training and rejects the old CPU backend', async ( assert ) => {

				const cpuTrainer = new NeuralAppearanceTrainer( {
					backend: 'cpu',
					resolution: 2,
					iterations: 1,
					batchSize: 4,
					hiddenSize: 4,
					yieldEvery: 0,
					seed: 1
				} );

				const gpuTrainer = new NeuralAppearanceTrainer( {
					backend: 'gpu',
					resolution: 2,
					iterations: 1,
					batchSize: 4,
					hiddenSize: 4
				} );

				await assert.rejects(
					cpuTrainer.train( { material: {} } ),
					/CPU backend training is no longer supported/,
					'rejects the removed CPU backend'
				);
				await assert.rejects(
					gpuTrainer.train( { material: {}, renderer: null } ),
					/WebGPU renderer is required for neural appearance training/,
					'requires a WebGPU renderer for training'
				);

			} );

		} );

	} );

} );

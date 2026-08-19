import {
	NeuralAppearanceTrainer,
	estimateTrainingMemory
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTrainer.js';
import {
	createGpuMaterialTeacher,
	NeuralAppearanceTeacherEvaluator
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceTrainer', () => {

			QUnit.test( 'estimates multiresolution grid memory', ( assert ) => {

				const memory = estimateTrainingMemory( 3, 2, 8 );

				assert.strictEqual( memory.levels, 3, 'reports the requested level count' );
				assert.deepEqual( memory.resolutions, [ 2, 4, 8 ], 'geometrically spaces resolutions between base and target' );
				assert.strictEqual( memory.latentTexels, 4 + 16 + 64, 'counts texels across independent grid levels' );
				assert.strictEqual( memory.trainingBytes, memory.latentTexels * 4 * 4 * 4, 'includes values, gradients, and two Adam moments' );
				assert.strictEqual( memory.exportBytes, memory.latentTexels * 4 * 2, 'estimates four FP16 latent channels per level' );

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
					levels: 2,
					baseResolution: 2,
					targetResolution: 4,
					iterations: 1,
					batchSize: 4,
					hiddenSize: 4,
					yieldEvery: 0,
					seed: 1
				} );

				const gpuTrainer = new NeuralAppearanceTrainer( {
					backend: 'gpu',
					levels: 2,
					baseResolution: 2,
					targetResolution: 4,
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

			QUnit.test( 'caches per-target-mode teacher resources instead of disposing/rebuilding on every mode switch', async ( assert ) => {

				const renderer = {
					isWebGPURenderer: true,
					toneMapping: 0,
					init: async () => {},
					getRenderTarget: () => null,
					getClearAlpha: () => 1,
					getClearColor() {},
					setClearColor() {},
					setRenderTarget() {},
					render() {},
					readRenderTargetPixelsAsync: async ( target, x, y, width, height ) => new Uint16Array( width * height * 4 )
				};

				const material = { isMeshPhysicalNodeMaterial: true, clone: () => material };
				const teacher = createGpuMaterialTeacher( material, renderer, { batchSize: 4, environment: {} } );

				let buildCount = 0;
				const originalCreateResources = teacher._createResources.bind( teacher );
				teacher._createResources = ( targetMode ) => {

					buildCount ++;
					return originalCreateResources( targetMode );

				};

				const sample = { uv: [ 0.5, 0.5 ], wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ], normal: [ 0, 0, 1 ], tangent: [ 1, 0, 0 ], bitangent: [ 0, 1, 0 ] };

				// 'opacity' (ungrouped) and 'iblQuery' (part of the 'iblProbe' MRT
				// group) exercise two independently-cached bundle keys without
				// touching the 'direct' (brdf-lit) group, which needs a real
				// WebGPU-registered PhysicalLightingModel this mock renderer/
				// material can't provide.
				await teacher.evaluateBatch( [ sample ], 'opacity' );
				await teacher.evaluateBatch( [ sample ], 'iblQuery' );
				await teacher.evaluateBatch( [ sample ], 'opacity' );
				await teacher.evaluateBatch( [ sample ], 'iblQuery' );

				assert.strictEqual( buildCount, 2, 'builds each target-mode bundle exactly once, on first use' );
				assert.strictEqual( teacher._modeBundles.size, 2, 'keeps both bundles cached simultaneously' );

			} );

			QUnit.test( 'merges related IBL-probe target modes into one MRT render+readback', async ( assert ) => {

				const renderCalls = [];
				const renderer = {
					isWebGPURenderer: true,
					toneMapping: 0,
					init: async () => {},
					getRenderTarget: () => null,
					getClearAlpha: () => 1,
					getClearColor() {},
					setClearColor() {},
					setRenderTarget() {},
					render() {

						renderCalls.push( true );

					},
					readRenderTargetPixelsAsync: async ( target, x, y, width, height ) => new Uint16Array( width * height * 4 )
				};

				const material = { isMeshPhysicalNodeMaterial: true, clone: () => material };
				const teacher = createGpuMaterialTeacher( material, renderer, { batchSize: 4, environment: {} } );

				const sample = { uv: [ 0.5, 0.5 ], wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ], normal: [ 0, 0, 1 ], tangent: [ 1, 0, 0 ], bitangent: [ 0, 1, 0 ] };
				const samples = [ sample ];

				await teacher.evaluateBatch( samples, 'iblQuery' );
				await teacher.evaluateBatch( samples, 'iblIncoming' );
				await teacher.evaluateBatch( samples, 'iblIrradiance' );

				assert.strictEqual( renderCalls.length, 1, 'renders the shared IBL probe group once for the same sample batch, not once per requested mode' );
				assert.strictEqual( teacher._modeBundles.size, 1, 'builds a single merged bundle for the whole iblProbe group' );

			} );

		} );

	} );

} );

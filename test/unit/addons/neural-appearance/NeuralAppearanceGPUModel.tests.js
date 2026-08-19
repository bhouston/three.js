import {
	computeModelLayout,
	NeuralAppearanceGPUModel
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceGPUModel.js';
import {
	createTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createAdamWeightsComputeNode,
	createAdamLatentsComputeNode
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceGPUComputeTSL.js';
import {
	createResetGradientNormComputeNode,
	createResetGradientsComputeNode
} from '../../../../examples/jsm/neural/NeuralGPUComputeTSL.js';
import { createModel } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';

const SMALL_GRID = { levels: 3, baseResolution: 2, targetResolution: 8 };

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceGPUModel', () => {

			QUnit.test( 'computes model layouts for various hidden sizes and output features', ( assert ) => {

				const baseLayout = computeModelLayout( {
					...SMALL_GRID,
					hiddenSize: 8,
					outputFeatures: { emission: false, opacity: false }
				} );

				// rotation: 16 * 12 = 192
				// layer 0: 28 * 8 = 224 weights, 8 biases
				// layer 1: 8 * 8 = 64 weights, 8 biases
				// layer 2: 8 * 3 = 24 weights, 3 biases
				// direct total = 192 + 224 + 8 + 64 + 8 + 24 + 3 = 523
				// IBL default hidden size is 16:
				// query 22-16-4: 22 * 16 + 16 + 16 * 4 + 4 = 436
				// each indirect probe 22-16-3: 22 * 16 + 16 + 16 * 3 + 3 = 419
				// iblWeightCount = 436 + 419 + 419 = 1274
				assert.strictEqual( baseLayout.rotationOffset, 0, 'rotation starts at offset 0' );
				assert.strictEqual( baseLayout.rotationCount, 192, 'rotation has 192 weights' );
				assert.strictEqual( baseLayout.layer0WeightsOffset, 192, 'layer 0 weights offset is 192' );
				assert.strictEqual( baseLayout.layer0WeightsCount, 224, 'layer 0 weights count is 224' );
				assert.strictEqual( baseLayout.layer0BiasesOffset, 416, 'layer 0 biases offset is 416' );
				assert.strictEqual( baseLayout.layer0BiasesCount, 8, 'layer 0 biases count is 8' );
				assert.strictEqual( baseLayout.layer1WeightsOffset, 424, 'layer 1 weights offset is 424' );
				assert.strictEqual( baseLayout.layer1WeightsCount, 64, 'layer 1 weights count is 64' );
				assert.strictEqual( baseLayout.layer1BiasesOffset, 488, 'layer 1 biases offset is 488' );
				assert.strictEqual( baseLayout.layer1BiasesCount, 8, 'layer 1 biases count is 8' );
				assert.strictEqual( baseLayout.layer2WeightsOffset, 496, 'layer 2 weights offset is 496' );
				assert.strictEqual( baseLayout.layer2WeightsCount, 24, 'layer 2 weights count is 24' );
				assert.strictEqual( baseLayout.layer2BiasesOffset, 520, 'layer 2 biases offset is 520' );
				assert.strictEqual( baseLayout.layer2BiasesCount, 3, 'layer 2 biases count is 3' );
				assert.strictEqual( baseLayout.directWeightCount, 523, 'direct weights exclude the IBL head' );
				assert.strictEqual( baseLayout.iblLayer0WeightsOffset, 523, 'IBL head starts after the BRDF weights' );
				assert.strictEqual( baseLayout.iblLayer0WeightsCount, 22 * 16, 'IBL layer 0 weights count uses default hidden size' );
				assert.strictEqual( baseLayout.iblLayer1WeightsCount, 16 * 4, 'IBL query layer 1 weights count uses 4 outputs' );
				assert.strictEqual( baseLayout.iblWeightCount, 1274, 'IBL weight count matches query 22-H-4 plus two indirect 22-H-3 heads' );
				assert.strictEqual( baseLayout.totalWeights, 1797, 'total weights matches sum of direct and IBL layers' );
				assert.strictEqual( baseLayout.sampleStride, 34, 'packs BRDF sample plus IBL query, probes, and isolate indirect labels' );

				// Multiresolution grid levels=3, baseResolution=2, targetResolution=8
				// => resolutions [2, 4, 8] => texel counts 4, 16, 64 => float counts (x4 channels) 16, 64, 256
				assert.strictEqual( baseLayout.gridLevels.length, 3, 'creates 3 grid levels (2x2, 4x4, 8x8)' );
				assert.strictEqual( baseLayout.gridLevels[ 0 ].offset, 0, 'level 0 starts at offset 0' );
				assert.strictEqual( baseLayout.gridLevels[ 0 ].floatCount, 16, 'level 0 has 16 floats' );
				assert.strictEqual( baseLayout.gridLevels[ 1 ].offset, 16, 'level 1 starts at offset 16' );
				assert.strictEqual( baseLayout.gridLevels[ 1 ].floatCount, 64, 'level 1 has 64 floats' );
				assert.strictEqual( baseLayout.gridLevels[ 2 ].offset, 80, 'level 2 starts at offset 80' );
				assert.strictEqual( baseLayout.gridLevels[ 2 ].floatCount, 256, 'level 2 has 256 floats' );
				assert.strictEqual( baseLayout.totalLatents, 336, 'total latents is 336 floats' );

				// With auxiliary heads
				const auxLayout = computeModelLayout( {
					...SMALL_GRID,
					hiddenSize: 16,
					outputFeatures: { emission: true, opacity: true }
				} );

				assert.strictEqual( auxLayout.supportsEmission, true, 'supports emission' );
				assert.strictEqual( auxLayout.supportsOpacity, true, 'supports opacity' );
				assert.ok( auxLayout.emissionWeightsOffset > 0, 'allocates emission weights' );
				assert.strictEqual( auxLayout.emissionWeightsCount, 48, 'emission weights count is 48' );
				assert.strictEqual( auxLayout.emissionBiasesCount, 3, 'emission biases count is 3' );
				assert.ok( auxLayout.opacityWeightsOffset > 0, 'allocates opacity weights' );
				assert.strictEqual( auxLayout.opacityWeightsCount, 16, 'opacity weights count is 16' );
				assert.strictEqual( auxLayout.opacityBiasesCount, 1, 'opacity biases count is 1' );
				assert.ok( auxLayout.iblLayer0WeightsOffset > auxLayout.opacityBiasesOffset, 'places IBL weights after auxiliary heads' );

			} );

			QUnit.test( 'initializes GPU storage buffers and synchronizes CPU model parameters', ( assert ) => {

				const random = () => 0.5;
				const options = {
					...SMALL_GRID,
					hiddenSize: 4,
					batchSize: 8,
					outputFeatures: { emission: true, opacity: true }
				};

				const cpuModel = createModel( options, random );
				const gpuModel = new NeuralAppearanceGPUModel( options );

				assert.ok( gpuModel.weightsStorage, 'creates weightsStorage node' );
				assert.ok( gpuModel.gradWeightsAtomic, 'creates gradWeightsAtomic node' );
				assert.ok( gpuModel.latentsStorage, 'creates latentsStorage node' );
				assert.ok( gpuModel.gradLatentsAtomic, 'creates gradLatentsAtomic node' );
				assert.ok( gpuModel.samplesStorage, 'creates samplesStorage node' );
				assert.ok( gpuModel.activationsStorage, 'creates activationsStorage node' );
				assert.ok( gpuModel.lossAtomic, 'creates lossAtomic node' );

				// Populate GPU buffers from CPU model
				gpuModel.initFromCPUModel( cpuModel );

				const weightsArray = gpuModel.weightsAttribute.array;
				const latentsArray = gpuModel.latentsAttribute.array;

				// Verify rotation weights copy
				for ( let i = 0; i < 192; i ++ ) {

					assert.strictEqual( weightsArray[ gpuModel.layout.rotationOffset + i ], cpuModel.rotationWeights[ i ], `rotation weight ${i} matches` );

				}

				// Verify decoder layer 0 weights copy
				const l0Weights = cpuModel.decoder.layers[ 0 ].weights;
				for ( let i = 0; i < l0Weights.length; i ++ ) {

					assert.strictEqual( weightsArray[ gpuModel.layout.layer0WeightsOffset + i ], l0Weights[ i ], `layer 0 weight ${i} matches` );

				}

				// Verify latents copy
				const level0 = cpuModel.latentGrids[ 0 ];
				for ( let i = 0; i < level0.data.length; i ++ ) {

					assert.strictEqual( latentsArray[ i ], level0.data[ i ], `latent float ${i} matches` );

				}

				assert.strictEqual( weightsArray[ gpuModel.layout.iblLayer1BiasesOffset ], cpuModel.iblHead.layers[ 1 ].biases[ 0 ], 'IBL head weight layout is initialized' );

			} );

			QUnit.test( 'serializes training samples and auxiliary targets to storage buffer', ( assert ) => {

				const options = {
					...SMALL_GRID,
					hiddenSize: 4,
					batchSize: 4
				};

				const gpuModel = new NeuralAppearanceGPUModel( options );
				const samples = [
					{
						uv: [ 0.25, 0.75 ],
						wi: [ 0, 0, 1 ],
						wo: [ 0, 1, 0 ],
						target: [ 0.5, 0.6, 0.7 ],
						weight: 2.0,
						emissionTarget: [ 0.1, 0.2, 0.3 ],
						opacityTarget: 0.8
					},
					{
						uv: [ 0.5, 0.5 ],
						wi: [ 1, 0, 0 ],
						wo: [ 0, 0, 1 ],
						target: [ 1, 1, 1 ],
						weight: 1.0
					}
				];

				gpuModel.uploadSamples( samples, 0.005, 10 );

				assert.strictEqual( gpuModel.learningRateUniform.value, 0.005, 'sets learning rate uniform' );
				assert.strictEqual( gpuModel.stepUniform.value, 10, 'sets step uniform' );
				assert.ok( gpuModel.invBatchUniform.value > 0, 'calculates inverse batch weight' );

				const data = gpuModel.samplesAttribute.array;
				const stride = gpuModel.layout.sampleStride;

				// Sample 0 assertions
				assert.strictEqual( data[ 0 ], 0.25, 'sample 0 uv.x' );
				assert.strictEqual( data[ 1 ], 0.75, 'sample 0 uv.y' );
				assert.strictEqual( data[ 2 ], 0, 'sample 0 wi.x' );
				assert.strictEqual( data[ 3 ], 0, 'sample 0 wi.y' );
				assert.strictEqual( data[ 4 ], 1, 'sample 0 wi.z' );
				assert.strictEqual( data[ 8 ], 2.0, 'sample 0 weight' );
				assert.ok( Math.abs( data[ 9 ] - 0.5 ) < 1e-6, 'sample 0 target.r' );
				assert.ok( Math.abs( data[ 10 ] - 0.6 ) < 1e-6, 'sample 0 target.g' );
				assert.ok( Math.abs( data[ 11 ] - 0.7 ) < 1e-6, 'sample 0 target.b' );
				assert.strictEqual( data[ 12 ], 1.0, 'sample 0 has emission' );
				assert.ok( Math.abs( data[ 13 ] - 0.1 ) < 1e-6, 'sample 0 emission.r' );
				assert.ok( Math.abs( data[ 16 ] - 0.8 ) < 1e-6, 'sample 0 opacity target' );

				// Sample 1 assertions
				assert.strictEqual( data[ stride + 0 ], 0.5, 'sample 1 uv.x' );
				assert.strictEqual( data[ stride + 1 ], 0.5, 'sample 1 uv.y' );
				assert.strictEqual( data[ stride + 12 ], 0.0, 'sample 1 no emission' );
				assert.strictEqual( data[ stride + 16 ], - 1.0, 'sample 1 no opacity' );
				assert.strictEqual( data[ 17 ], 0, 'sample 0 default IBL target is zero-filled' );

			} );

			QUnit.test( 'rejects oversized sample batches instead of truncating GPU training data', ( assert ) => {

				const gpuModel = new NeuralAppearanceGPUModel( {
					...SMALL_GRID,
					hiddenSize: 4,
					batchSize: 2
				} );
				const sample = {
					uv: [ 0.5, 0.5 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					target: [ 1, 1, 1 ],
					weight: 1
				};

				assert.throws(
					() => gpuModel.uploadSamples( [ sample, sample, sample ] ),
					/Batch contains 3 samples, but GPU buffers were allocated for 2/,
					'does not silently drop samples beyond the allocated dispatch size'
				);

			} );

			QUnit.test( 'creates TSL compute nodes for forward/backward and Adam passes', ( assert ) => {

				const options = {
					...SMALL_GRID,
					hiddenSize: 8,
					batchSize: 16,
					outputFeatures: { emission: true, opacity: true }
				};

				const gpuModel = new NeuralAppearanceGPUModel( options );
				const trainBatchNode = createTrainBatchComputeNode( gpuModel );
				const resetGradientNormNode = createResetGradientNormComputeNode( gpuModel );
				const resetGradientsNode = createResetGradientsComputeNode( gpuModel );
				const accumulateGradientNormNode = createAccumulateGradientNormComputeNode( gpuModel );
				const adamWeightsNode = createAdamWeightsComputeNode( gpuModel );
				const adamLatentsNode = createAdamLatentsComputeNode( gpuModel );

				assert.ok( trainBatchNode && trainBatchNode.isComputeNode, 'creates train batch compute node' );
				assert.ok( resetGradientNormNode && resetGradientNormNode.isComputeNode, 'creates gradient norm reset compute node' );
				assert.ok( resetGradientsNode && resetGradientsNode.isComputeNode, 'creates gradient reset compute node' );
				assert.ok( accumulateGradientNormNode && accumulateGradientNormNode.isComputeNode, 'creates gradient norm accumulation compute node' );
				assert.ok( adamWeightsNode && adamWeightsNode.isComputeNode, 'creates Adam weights compute node' );
				assert.ok( adamLatentsNode && adamLatentsNode.isComputeNode, 'creates Adam latents compute node' );
				assert.strictEqual( trainBatchNode.count, 16, 'train batch dispatch count matches batch size' );
				assert.strictEqual( resetGradientNormNode.count, 1, 'gradient norm reset dispatches one invocation' );
				assert.strictEqual( resetGradientsNode.count, gpuModel.layout.totalWeights + gpuModel.layout.totalLatents, 'gradient reset covers all gradients' );
				assert.strictEqual( accumulateGradientNormNode.count, gpuModel.layout.totalWeights + gpuModel.layout.totalLatents, 'gradient norm accumulation covers all trainable values' );
				assert.strictEqual( adamWeightsNode.count, gpuModel.layout.totalWeights, 'Adam weights dispatch count matches total weights' );
				assert.strictEqual( adamLatentsNode.count, gpuModel.layout.totalLatents, 'Adam latents dispatch count matches total latents' );

			} );

			QUnit.test( 'syncToCPU correctly updates CPU model weights and multi-level latents', async ( assert ) => {

				const random = () => 0.25;
				const options = {
					...SMALL_GRID,
					hiddenSize: 4,
					batchSize: 4,
					outputFeatures: { emission: true, opacity: true }
				};

				const cpuModel = createModel( options, random );
				const gpuModel = new NeuralAppearanceGPUModel( options );

				// Mock renderer with getArrayBufferAsync
				const mockRenderer = {
					isWebGPURenderer: true,
					async getArrayBufferAsync( attribute ) {

						return attribute.array.buffer.slice( 0 );

					}
				};

				// Modify GPU weights
				const weightsArray = gpuModel.weightsAttribute.array;
				weightsArray.fill( 0.75 );

				// Modify GPU latents
				const latentsArray = gpuModel.latentsAttribute.array;
				latentsArray.fill( 0.42 );

				await gpuModel.syncToCPU( cpuModel, mockRenderer );

				assert.strictEqual( cpuModel.rotationWeights[ 0 ], 0.75, 'synced rotation weight' );
				assert.strictEqual( cpuModel.decoder.layers[ 0 ].weights[ 0 ], 0.75, 'synced decoder layer 0 weight' );
				assert.strictEqual( cpuModel.decoder.layers[ 1 ].weights[ 0 ], 0.75, 'synced decoder layer 1 weight' );
				assert.strictEqual( cpuModel.decoder.layers[ 2 ].weights[ 0 ], 0.75, 'synced decoder layer 2 weight' );
				assert.strictEqual( cpuModel.iblHead.layers[ 0 ].weights[ 0 ], 0.75, 'synced IBL layer 0 weight' );
				assert.strictEqual( cpuModel.emissionHead.layers[ 0 ].weights[ 0 ], 0.75, 'synced emission head weight' );
				assert.strictEqual( cpuModel.opacityHead.layers[ 0 ].weights[ 0 ], 0.75, 'synced opacity head weight' );

				assert.ok( Math.abs( cpuModel.latentGrids[ 0 ].data[ 0 ] - 0.42 ) < 1e-6, 'synced level 0 latents' );
				assert.ok( Math.abs( cpuModel.latentGrids[ 1 ].data[ 0 ] - 0.42 ) < 1e-6, 'synced level 1 latents' );

			} );

			QUnit.test( 'reads back and resets batch loss accumulator', async ( assert ) => {

				const options = { ...SMALL_GRID, hiddenSize: 4, batchSize: 4 };
				const gpuModel = new NeuralAppearanceGPUModel( options );

				// Simulate GPU loss atomic accumulation (e.g., loss = 0.35 * FIXED_POINT_SCALE)
				gpuModel.lossAttribute.array[ 0 ] = 35000;
				gpuModel.lossAttribute.array[ 1 ] = 20000;
				gpuModel.lossAttribute.array[ 2 ] = 15000;

				const mockRenderer = {
					isWebGPURenderer: true,
					async getArrayBufferAsync( attribute ) {

						return attribute.array.buffer.slice( 0 );

					}
				};

				const loss = await gpuModel.readLoss( mockRenderer );
				assert.ok( Math.abs( loss - 0.35 ) < 1e-5, 'loss decoded correctly from fixed point' );
				assert.strictEqual( gpuModel.lossAttribute.array[ 0 ], 0, 'resets total loss accumulator' );
				assert.strictEqual( gpuModel.lossAttribute.array[ 1 ], 0, 'resets direct loss accumulator' );
				assert.strictEqual( gpuModel.lossAttribute.array[ 2 ], 0, 'resets IBL loss accumulator' );

			} );

		} );

	} );

} );

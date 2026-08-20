import { describe, test, expect } from 'vitest';
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

const SMALL_GRID = { levels: 3, baseResolution: 2, growthFactor: 2 };

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceGPUModel', () => {

			test( 'computes model layouts for various hidden sizes and output features', () => {

				const baseLayout = computeModelLayout( {
					...SMALL_GRID,
					hiddenSize: 8,
					outputFeatures: { emission: false, opacity: false }
				} );

				// SMALL_GRID has levels: 3, so latentChannels = 3 * 4 = 12:
				// rotation: 12 * 12 = 144
				// layer 0: (12 + 12) * 8 = 192 weights, 8 biases
				// layer 1: 8 * 8 = 64 weights, 8 biases
				// layer 2: 8 * 3 = 24 weights, 3 biases
				// direct total = 144 + 192 + 8 + 64 + 8 + 24 + 3 = 443
				// IBL default hidden size is 16, iblInputSize = 12 + 6 = 18:
				// query 18-16-4: 18 * 16 + 16 + 16 * 4 + 4 = 372
				// each indirect probe 18-16-3: 18 * 16 + 16 + 16 * 3 + 3 = 355
				// iblWeightCount = 372 + 355 + 355 = 1082
				expect( baseLayout.rotationOffset ).toBe( 0 );
				expect( baseLayout.rotationCount ).toBe( 144 );
				expect( baseLayout.layer0WeightsOffset ).toBe( 144 );
				expect( baseLayout.layer0WeightsCount ).toBe( 192 );
				expect( baseLayout.layer0BiasesOffset ).toBe( 336 );
				expect( baseLayout.layer0BiasesCount ).toBe( 8 );
				expect( baseLayout.layer1WeightsOffset ).toBe( 344 );
				expect( baseLayout.layer1WeightsCount ).toBe( 64 );
				expect( baseLayout.layer1BiasesOffset ).toBe( 408 );
				expect( baseLayout.layer1BiasesCount ).toBe( 8 );
				expect( baseLayout.layer2WeightsOffset ).toBe( 416 );
				expect( baseLayout.layer2WeightsCount ).toBe( 24 );
				expect( baseLayout.layer2BiasesOffset ).toBe( 440 );
				expect( baseLayout.layer2BiasesCount ).toBe( 3 );
				expect( baseLayout.directWeightCount ).toBe( 443 );
				expect( baseLayout.iblLayer0WeightsOffset ).toBe( 443 );
				expect( baseLayout.iblLayer0WeightsCount ).toBe( baseLayout.iblInputSize * 16 );
				expect( baseLayout.iblLayer1WeightsCount ).toBe( 16 * 4 );
				expect( baseLayout.iblWeightCount ).toBe( 1082 );
				expect( baseLayout.totalWeights ).toBe( 1525 );
				expect( baseLayout.sampleStride ).toBe( 34 );

				// Multiresolution grid levels=3, baseResolution=2, growthFactor=2
				// => resolutions [2, 4, 8] => texel counts 4, 16, 64 => float counts (x4 channels) 16, 64, 256
				expect( baseLayout.gridLevels.length ).toBe( 3 );
				expect( baseLayout.gridLevels[ 0 ].offset ).toBe( 0 );
				expect( baseLayout.gridLevels[ 0 ].floatCount ).toBe( 16 );
				expect( baseLayout.gridLevels[ 1 ].offset ).toBe( 16 );
				expect( baseLayout.gridLevels[ 1 ].floatCount ).toBe( 64 );
				expect( baseLayout.gridLevels[ 2 ].offset ).toBe( 80 );
				expect( baseLayout.gridLevels[ 2 ].floatCount ).toBe( 256 );
				expect( baseLayout.totalLatents ).toBe( 336 );

				// With auxiliary heads
				const auxLayout = computeModelLayout( {
					...SMALL_GRID,
					hiddenSize: 16,
					outputFeatures: { emission: true, opacity: true }
				} );

				expect( auxLayout.supportsEmission ).toBe( true );
				expect( auxLayout.supportsOpacity ).toBe( true );
				expect( auxLayout.emissionWeightsOffset > 0 ).toBeTruthy();
				expect( auxLayout.emissionWeightsCount ).toBe( 36 );
				expect( auxLayout.emissionBiasesCount ).toBe( 3 );
				expect( auxLayout.opacityWeightsOffset > 0 ).toBeTruthy();
				expect( auxLayout.opacityWeightsCount ).toBe( 12 );
				expect( auxLayout.opacityBiasesCount ).toBe( 1 );
				expect( auxLayout.iblLayer0WeightsOffset > auxLayout.opacityBiasesOffset ).toBeTruthy();

			} );

			test( 'initializes GPU storage buffers and synchronizes CPU model parameters', () => {

				const random = () => 0.5;
				const options = {
					...SMALL_GRID,
					hiddenSize: 4,
					batchSize: 8,
					outputFeatures: { emission: true, opacity: true }
				};

				const cpuModel = createModel( options, random );
				const gpuModel = new NeuralAppearanceGPUModel( options );

				expect( gpuModel.weightsBuffers.valuesStorage ).toBeTruthy();
				expect( gpuModel.weightsBuffers.gradAtomic ).toBeTruthy();
				expect( gpuModel.latentsBuffers.valuesStorage ).toBeTruthy();
				expect( gpuModel.latentsBuffers.gradAtomic ).toBeTruthy();
				expect( gpuModel.samplesStorage ).toBeTruthy();
				expect( gpuModel.activationsStorage ).toBeTruthy();
				expect( gpuModel.lossAtomic ).toBeTruthy();

				// Populate GPU buffers from CPU model
				gpuModel.initFromCPUModel( cpuModel );

				const weightsArray = gpuModel.weightsBuffers.attribute.array;
				const latentsArray = gpuModel.latentsBuffers.attribute.array;

				// Verify rotation weights copy
				for ( let i = 0; i < gpuModel.layout.rotationCount; i ++ ) {

					expect( weightsArray[ gpuModel.layout.rotationOffset + i ] ).toBe( cpuModel.rotationWeights[ i ] );

				}

				// Verify decoder layer 0 weights copy
				const l0Weights = cpuModel.decoder.layers[ 0 ].weights;
				for ( let i = 0; i < l0Weights.length; i ++ ) {

					expect( weightsArray[ gpuModel.layout.layer0WeightsOffset + i ] ).toBe( l0Weights[ i ] );

				}

				// Verify latents copy
				const level0 = cpuModel.latentGrids[ 0 ];
				for ( let i = 0; i < level0.data.length; i ++ ) {

					expect( latentsArray[ i ] ).toBe( level0.data[ i ] );

				}

				expect( weightsArray[ gpuModel.layout.iblLayer1BiasesOffset ] ).toBe( cpuModel.iblHead.layers[ 1 ].biases[ 0 ] );

			} );

			test( 'serializes training samples and auxiliary targets to storage buffer', () => {

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

				expect( gpuModel.learningRateUniform.value ).toBe( 0.005 );
				expect( gpuModel.stepUniform.value ).toBe( 10 );
				expect( gpuModel.invBatchUniform.value > 0 ).toBeTruthy();

				const data = gpuModel.samplesAttribute.array;
				const stride = gpuModel.layout.sampleStride;

				// Sample 0 assertions
				expect( data[ 0 ] ).toBe( 0.25 );
				expect( data[ 1 ] ).toBe( 0.75 );
				expect( data[ 2 ] ).toBe( 0 );
				expect( data[ 3 ] ).toBe( 0 );
				expect( data[ 4 ] ).toBe( 1 );
				expect( data[ 8 ] ).toBe( 2.0 );
				expect( Math.abs( data[ 9 ] - 0.5 ) < 1e-6 ).toBeTruthy();
				expect( Math.abs( data[ 10 ] - 0.6 ) < 1e-6 ).toBeTruthy();
				expect( Math.abs( data[ 11 ] - 0.7 ) < 1e-6 ).toBeTruthy();
				expect( data[ 12 ] ).toBe( 1.0 );
				expect( Math.abs( data[ 13 ] - 0.1 ) < 1e-6 ).toBeTruthy();
				expect( Math.abs( data[ 16 ] - 0.8 ) < 1e-6 ).toBeTruthy();

				// Sample 1 assertions
				expect( data[ stride + 0 ] ).toBe( 0.5 );
				expect( data[ stride + 1 ] ).toBe( 0.5 );
				expect( data[ stride + 12 ] ).toBe( 0.0 );
				expect( data[ stride + 16 ] ).toBe( - 1.0 );
				expect( data[ 17 ] ).toBe( 0 );

			} );

			test( 'rejects oversized sample batches instead of truncating GPU training data', () => {

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

				expect( () => gpuModel.uploadSamples( [ sample, sample, sample ] ) ).toThrow(
					/Batch contains 3 samples, but GPU buffers were allocated for 2/
				);

			} );

			test( 'creates TSL compute nodes for forward/backward and Adam passes', () => {

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

				expect( trainBatchNode && trainBatchNode.isComputeNode ).toBeTruthy();
				expect( resetGradientNormNode && resetGradientNormNode.isComputeNode ).toBeTruthy();
				expect( resetGradientsNode && resetGradientsNode.isComputeNode ).toBeTruthy();
				expect( accumulateGradientNormNode && accumulateGradientNormNode.isComputeNode ).toBeTruthy();
				expect( adamWeightsNode && adamWeightsNode.isComputeNode ).toBeTruthy();
				expect( adamLatentsNode && adamLatentsNode.isComputeNode ).toBeTruthy();
				expect( trainBatchNode.count ).toBe( 16 );
				expect( resetGradientNormNode.count ).toBe( 1 );
				expect( resetGradientsNode.count ).toBe( gpuModel.layout.totalWeights + gpuModel.layout.totalLatents );
				expect( accumulateGradientNormNode.count ).toBe( gpuModel.layout.totalWeights + gpuModel.layout.totalLatents );
				expect( adamWeightsNode.count ).toBe( gpuModel.layout.totalWeights );
				expect( adamLatentsNode.count ).toBe( gpuModel.layout.totalLatents );

			} );

			test( 'syncToCPU correctly updates CPU model weights and multi-level latents', async () => {

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
				const weightsArray = gpuModel.weightsBuffers.attribute.array;
				weightsArray.fill( 0.75 );

				// Modify GPU latents
				const latentsArray = gpuModel.latentsBuffers.attribute.array;
				latentsArray.fill( 0.42 );

				await gpuModel.syncToCPU( cpuModel, mockRenderer );

				expect( cpuModel.rotationWeights[ 0 ] ).toBe( 0.75 );
				expect( cpuModel.decoder.layers[ 0 ].weights[ 0 ] ).toBe( 0.75 );
				expect( cpuModel.decoder.layers[ 1 ].weights[ 0 ] ).toBe( 0.75 );
				expect( cpuModel.decoder.layers[ 2 ].weights[ 0 ] ).toBe( 0.75 );
				expect( cpuModel.iblHead.layers[ 0 ].weights[ 0 ] ).toBe( 0.75 );
				expect( cpuModel.emissionHead.layers[ 0 ].weights[ 0 ] ).toBe( 0.75 );
				expect( cpuModel.opacityHead.layers[ 0 ].weights[ 0 ] ).toBe( 0.75 );

				expect( Math.abs( cpuModel.latentGrids[ 0 ].data[ 0 ] - 0.42 ) < 1e-6 ).toBeTruthy();
				expect( Math.abs( cpuModel.latentGrids[ 1 ].data[ 0 ] - 0.42 ) < 1e-6 ).toBeTruthy();

			} );

			test( 'reads back and resets batch loss accumulator', async () => {

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
				expect( Math.abs( loss - 0.35 ) < 1e-5 ).toBeTruthy();
				expect( gpuModel.lossAttribute.array[ 0 ] ).toBe( 0 );
				expect( gpuModel.lossAttribute.array[ 1 ] ).toBe( 0 );
				expect( gpuModel.lossAttribute.array[ 2 ] ).toBe( 0 );

			} );

		} );

	} );

} );

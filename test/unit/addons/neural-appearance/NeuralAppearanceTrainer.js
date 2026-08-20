import { describe, test, expect } from 'vitest';
import {
	NeuralAppearanceTrainer,
	estimateTrainingMemory
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTrainer.js';
import {
	createGpuMaterialTeacher,
	NeuralAppearanceTeacherEvaluator
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js';

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceTrainer', () => {

			test( 'estimates multiresolution grid memory', () => {

				const memory = estimateTrainingMemory( 3, 2, 8 );

				expect( memory.levels ).toBe( 3 );
				expect( memory.resolutions ).toEqual( [ 2, 4, 8 ] );
				expect( memory.latentTexels ).toBe( 4 + 16 + 64 );
				expect( memory.trainingBytes ).toBe( memory.latentTexels * 4 * 4 * 4 );
				expect( memory.exportBytes ).toBe( memory.latentTexels * 4 * 2 );

			} );

			test( 'requires training-safe output and gradient settings', async () => {

				await expect( new NeuralAppearanceTrainer( { outputActivation: { type: 'exp' } } ).train( { material: {} } ) ).rejects.toThrow(
					/Only linear output activation/
				);
				await expect( new NeuralAppearanceTrainer( { maxGradientNorm: Infinity } ).train( { material: {} } ) ).rejects.toThrow(
					/maxGradientNorm must be finite/
				);

			} );

			test( 'rejects unsupported GPU teacher materials', () => {

				expect( () => createGpuMaterialTeacher( { type: 'MeshBasicNodeMaterial' }, null ) ).toThrow(
					/supported MeshPhysicalNodeMaterial teacher is required/
				);
				expect(
					createGpuMaterialTeacher( { isMeshPhysicalNodeMaterial: true }, null ) instanceof NeuralAppearanceTeacherEvaluator
				).toBeTruthy();

			} );

			test( 'requires WebGPU training and rejects the old CPU backend', async () => {

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

				await expect( cpuTrainer.train( { material: {} } ) ).rejects.toThrow(
					/CPU backend training is no longer supported/
				);
				await expect( gpuTrainer.train( { material: {}, renderer: null } ) ).rejects.toThrow(
					/WebGPU renderer is required for neural appearance training/
				);

			} );

			test( 'caches per-target-mode teacher resources instead of disposing/rebuilding on every mode switch', async () => {

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

				expect( buildCount ).toBe( 2 );
				expect( teacher._modeBundles.size ).toBe( 2 );

			} );

			test( 'merges related IBL-probe target modes into one MRT render+readback', async () => {

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
				await teacher.evaluateBatch( samples, 'iblIndirectRadiance' );
				await teacher.evaluateBatch( samples, 'iblIndirectIrradiance' );

				// Two render calls, not five: 'iblProbe' (query/incoming/irradiance)
				// and 'iblIndirect' (indirectRadiance/indirectIrradiance) are each
				// their own MRT group -- kept as two groups rather than one merged
				// 5-attachment group to stay under the 32-bytes/sample MRT budget
				// some real WebGPU devices enforce (see GROUP_BY_MODE's header
				// comment in NeuralAppearanceTeacherEvaluator.js).
				expect( renderCalls.length ).toBe( 2 );
				expect( teacher._modeBundles.size ).toBe( 2 );

			} );

		} );

	} );

} );

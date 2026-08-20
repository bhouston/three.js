import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { NeuralAppearanceTrainer, createGpuMaterialTeacher } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTrainer.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// End-to-end (real GPU training, not just the CPU-side math) regression test
// for the "grid levels" bug documented in test/vitest/unit/neural-appearance/
// NeuralAppearanceModel.levels-bug.test.js: NeuralAppearanceFormat.js's
// LATENT_CHANNELS/DECODER_INPUT_SIZE/IBL_INPUT_SIZE/INDIRECT_INPUT_SIZE were
// fixed constants baked from the hardcoded default LEVELS=4, decoupled from
// whatever `levels` a caller actually configured - which is exactly what
// `webgpu_materials_neural_appearance.html`'s "grid levels" GUI control
// (offering 2/3/4/5/6/8) does. Every prediction came out NaN for any
// non-default choice - a total, silent training failure.
//
// The CPU-model-math half of the fix is covered by the unit test above; this
// file exercises the GPU half specifically (NeuralAppearanceGPUModel.js's
// buffer layout and NeuralAppearanceGPUComputeTSL.js's training kernels,
// both of which had the exact same class of bug), by actually training
// through the public NeuralAppearanceTrainer API with non-default `levels`
// values and asserting the loss stays finite.
describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceTrainer "grid levels" regression (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	function buildTeacherMaterial() {

		return new THREE.MeshPhysicalNodeMaterial( { color: 0x8899aa, roughness: 0.4, metalness: 0.1 } );

	}

	it.each( [ 2, 3, 5, 6, 8 ] )( 'trains to a finite, non-NaN loss with levels=%i (previously NaN from iteration 1)', async ( levels ) => {

		const material = buildTeacherMaterial();
		const teacher = createGpuMaterialTeacher( material, renderer, { batchSize: 256 } );

		const trainer = new NeuralAppearanceTrainer( {
			levels,
			baseResolution: 4,
			growthFactor: 2,
			hiddenSize: 8,
			iblHiddenSize: 8,
			batchSize: 256,
			iterations: 12,
			iblIterations: 0,
			learningRate: 0.001,
			seed: 1
		} );

		const result = await trainer.train( { material, renderer, teacher } );

		expect( result.iteration ).toBeGreaterThan( 0 );
		expect( Number.isFinite( result.loss ) ).toBe( true );
		expect( Number.isFinite( result.validationLoss ) ).toBe( true );
		expect( result.json.latents.levels.length ).toBe( levels );

		// The manifest's own declared input width must match this model's
		// real levels, not the fixed default-levels constant.
		expect( result.json.outputs.brdf.inputSize ).toBe( levels * 4 + 12 );

	}, 60000 );

} );

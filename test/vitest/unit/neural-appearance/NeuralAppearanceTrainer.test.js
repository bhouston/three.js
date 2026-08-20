import { beforeEach, describe, expect, it, vi } from 'vitest';

// NeuralAppearanceTrainer.js unconditionally imports several GPU-side
// modules (NeuralAppearanceTeacherEvaluator.js, NeuralAppearanceGPUModel.js,
// NeuralAppearanceGPUComputeTSL.js, ../neural/NeuralGPUComputeTSL.js) that
// themselves import 'three/tsl' / 'three/webgpu'. In this project's plain
// Node environment (no `three` -> build/three.webgpu.js alias - see
// vitest.config.js's comment on the 'browser' project's alias block) merely
// *importing* those files throws at module-load time, before any test here
// even runs. They're mocked out below so only the pure orchestration logic
// in NeuralAppearanceTrainer.js itself - validateTrainingSettings(), the
// outputFeatures/opacityMode/alphaCutoff derivation, and the abort() state
// machine - is under test; none of this exercises a real renderer or GPU
// compute kernel (that's what the `browser` project's
// NeuralAppearanceTrainer.perf.test.js and .levels-bug.test.js already do).
vi.mock( '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js', () => ( {
	createGpuMaterialTeacher: vi.fn()
} ) );

vi.mock( '../../../../examples/jsm/neural-appearance/NeuralAppearanceGPUModel.js', () => ( {
	NeuralAppearanceGPUModel: vi.fn( function () {

		this.layout = { directWeightCount: 0, iblWeightCount: 0 };
		this.initFromCPUModel = vi.fn();
		this.resetLoss = vi.fn();
		this.uploadSamples = vi.fn();
		this.syncToCPU = vi.fn( async () => {} );
		this.readLosses = vi.fn( async () => ( { loss: 0, directLoss: 0, iblLoss: 0 } ) );

	} )
} ) );

vi.mock( '../../../../examples/jsm/neural-appearance/NeuralAppearanceGPUComputeTSL.js', () => ( {
	createTrainBatchComputeNode: vi.fn( () => ( {} ) ),
	createAccumulateGradientNormComputeNode: vi.fn( () => ( {} ) ),
	createAdamWeightsComputeNode: vi.fn( () => ( {} ) ),
	createAdamLatentsComputeNode: vi.fn( () => ( {} ) )
} ) );

vi.mock( '../../../../examples/jsm/neural/NeuralGPUComputeTSL.js', () => ( {
	createResetGradientNormComputeNode: vi.fn( () => ( {} ) ),
	createResetGradientsComputeNode: vi.fn( () => ( {} ) )
} ) );

vi.mock( '../../../../examples/jsm/neural-appearance/NeuralAppearanceSampler.js', () => ( {
	generateTrainingSamples: vi.fn( async () => [] ),
	generateIBLTrainingSamples: vi.fn( async () => [] ),
	generateValidationSamples: vi.fn( async () => [] ),
	normalizeDirectLightingTargets: vi.fn()
} ) );

vi.mock( '../../../../examples/jsm/neural-appearance/NeuralAppearanceManifest.js', () => ( {
	createNeuralAppearanceManifest: vi.fn( () => ( { outputs: {} } ) ),
	exportNeuralAppearance: vi.fn( async () => ( {} ) )
} ) );

vi.mock( '../../../../examples/jsm/neural-appearance/NeuralAppearanceValidator.js', () => ( {
	evaluateRuntimeValidation: vi.fn( () => ( { loss: 0 } ) )
} ) );

const { createModel } = await import( '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js' );
const { NeuralAppearanceTrainer } = await import( '../../../../examples/jsm/neural-appearance/NeuralAppearanceTrainer.js' );

vi.mock( '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js', async ( importOriginal ) => {

	const actual = await importOriginal();
	return {
		...actual,
		createModel: vi.fn( actual.createModel )
	};

} );

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceTrainer (orchestration, mocked GPU)', () => {

	describe( 'validateTrainingSettings error paths (exercised via train())', () => {

		it( 'rejects when backend is explicitly "cpu"', async () => {

			const trainer = new NeuralAppearanceTrainer( { backend: 'cpu' } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: CPU backend training is no longer supported. Use a WebGPU renderer for training.'
			);

		} );

		it.each( [ 0, 1.5, -1, NaN ] )( 'rejects when levels is not a positive integer (levels=%p)', async ( levels ) => {

			const trainer = new NeuralAppearanceTrainer( { levels } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: levels must be a positive integer.'
			);

		} );

		it.each( [ 0, 2.5, -3, NaN ] )( 'rejects when baseResolution is not a positive integer (baseResolution=%p)', async ( baseResolution ) => {

			const trainer = new NeuralAppearanceTrainer( { baseResolution } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: baseResolution must be a positive integer.'
			);

		} );

		it( 'rejects when targetResolution is less than baseResolution', async () => {

			const trainer = new NeuralAppearanceTrainer( { baseResolution: 16, targetResolution: 8 } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: targetResolution must be an integer at least baseResolution.'
			);

		} );

		it( 'rejects when targetResolution is not an integer, even if numerically >= baseResolution', async () => {

			const trainer = new NeuralAppearanceTrainer( { baseResolution: 8, targetResolution: 8.5 } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: targetResolution must be an integer at least baseResolution.'
			);

		} );

		it.each( [ 'iterations', 'batchSize', 'hiddenSize' ] )( 'rejects when %s is not a positive integer', async ( name ) => {

			const trainer = new NeuralAppearanceTrainer( { [ name ]: 0 } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				`THREE.NeuralAppearanceTrainer: ${ name } must be a positive integer.`
			);

		} );

		it.each( [ -1, 1.5, NaN ] )( 'rejects when iblIterations is provided but not a non-negative integer (iblIterations=%p)', async ( iblIterations ) => {

			const trainer = new NeuralAppearanceTrainer( { iblIterations } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: iblIterations must be a non-negative integer.'
			);

		} );

		it( 'accepts iblIterations = 0 (a valid non-negative integer) and proceeds past validation', async () => {

			// No renderer is supplied, so this still rejects - but with the
			// *later* "renderer is required" error, proving iblIterations: 0
			// did not trip validateTrainingSettings.
			const trainer = new NeuralAppearanceTrainer( { iblIterations: 0 } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: WebGPU renderer is required for neural appearance training.'
			);

		} );

		it.each( [ -0.1, NaN, Infinity ] )( 'rejects when iblTrainingRatio is not finite and non-negative (iblTrainingRatio=%p)', async ( iblTrainingRatio ) => {

			const trainer = new NeuralAppearanceTrainer( { iblTrainingRatio } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: iblTrainingRatio must be finite and non-negative.'
			);

		} );

		it.each( [ 0, -0.1, NaN, Infinity ] )( 'rejects when iblLearningRateScale is not finite and > 0 (iblLearningRateScale=%p)', async ( iblLearningRateScale ) => {

			const trainer = new NeuralAppearanceTrainer( { iblLearningRateScale } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: iblLearningRateScale must be finite and greater than zero.'
			);

		} );

		it.each( [
			null,
			undefined,
			{ type: 'relu' },
			{ type: 'softplus' }
		] )( 'rejects unless outputActivation.type is exactly "linear" (outputActivation=%p)', async ( outputActivation ) => {

			const trainer = new NeuralAppearanceTrainer( { outputActivation } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: Only linear output activation is supported during training.'
			);

		} );

		it.each( [ 0, -1, NaN, Infinity ] )( 'rejects when maxGradientNorm is not finite and > 0 (maxGradientNorm=%p)', async ( maxGradientNorm ) => {

			const trainer = new NeuralAppearanceTrainer( { maxGradientNorm } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: maxGradientNorm must be finite and greater than zero.'
			);

		} );

		it.each( [ -0.1, 1.1, NaN ] )( 'rejects when minimumTrainingCosine is outside [0, 1] (minimumTrainingCosine=%p)', async ( minimumTrainingCosine ) => {

			const trainer = new NeuralAppearanceTrainer( { minimumTrainingCosine } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: minimumTrainingCosine must be between zero and one.'
			);

		} );

		it.each( [ 0, 1 ] )( 'accepts the inclusive boundary values of minimumTrainingCosine (minimumTrainingCosine=%p)', async ( minimumTrainingCosine ) => {

			const trainer = new NeuralAppearanceTrainer( { minimumTrainingCosine } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: WebGPU renderer is required for neural appearance training.'
			);

		} );

		it.each( [ -0.1, NaN, Infinity ] )( 'rejects when highlightLossScale is not finite and non-negative (highlightLossScale=%p)', async ( highlightLossScale ) => {

			const trainer = new NeuralAppearanceTrainer( { highlightLossScale } );
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: highlightLossScale must be finite and non-negative.'
			);

		} );

		it( 'accepts the default options untouched (proves the defaults themselves are valid) and proceeds past validation', async () => {

			const trainer = new NeuralAppearanceTrainer();
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: WebGPU renderer is required for neural appearance training.'
			);

		} );

	} );

	describe( 'renderer requirement (checked after validateTrainingSettings)', () => {

		it( 'rejects when no renderer is supplied', async () => {

			const trainer = new NeuralAppearanceTrainer();
			await expect( trainer.train( {} ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: WebGPU renderer is required for neural appearance training.'
			);

		} );

		it( 'rejects when the supplied renderer is not a WebGPURenderer (isWebGPURenderer !== true)', async () => {

			const trainer = new NeuralAppearanceTrainer();
			await expect( trainer.train( { renderer: {} } ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: WebGPU renderer is required for neural appearance training.'
			);
			await expect( trainer.train( { renderer: { isWebGPURenderer: false } } ) ).rejects.toThrow(
				'THREE.NeuralAppearanceTrainer: WebGPU renderer is required for neural appearance training.'
			);

		} );

	} );

	describe( 'abort() state machine (pure flag, no in-flight GPU train() involved)', () => {

		it( 'starts with abort not requested', () => {

			const trainer = new NeuralAppearanceTrainer();
			expect( trainer._abortRequested ).toBe( false );

		} );

		it( 'sets the internal abort flag and does not throw, even with no train() ever called', () => {

			const trainer = new NeuralAppearanceTrainer();
			expect( () => trainer.abort() ).not.toThrow();
			expect( trainer._abortRequested ).toBe( true );

		} );

		it( 'is idempotent - calling it repeatedly leaves the flag simply true', () => {

			const trainer = new NeuralAppearanceTrainer();
			trainer.abort();
			trainer.abort();
			trainer.abort();
			expect( trainer._abortRequested ).toBe( true );

		} );

		it( 'a pending abort survives a train() call that is rejected by validateTrainingSettings (the reset happens after validation)', async () => {

			const trainer = new NeuralAppearanceTrainer( { levels: 0 } ); // invalid: fails validation
			trainer.abort();
			expect( trainer._abortRequested ).toBe( true );

			await expect( trainer.train( {} ) ).rejects.toThrow( 'levels must be a positive integer' );

			// validateTrainingSettings() throws before the `this._abortRequested = false`
			// reset line runs, so a pending abort from before the call is untouched.
			expect( trainer._abortRequested ).toBe( true );

		} );

		it( 'a pending abort is cleared by a train() call that passes validation, even though it goes on to reject for lacking a renderer', async () => {

			const trainer = new NeuralAppearanceTrainer(); // valid default settings
			trainer.abort();
			expect( trainer._abortRequested ).toBe( true );

			await expect( trainer.train( {} ) ).rejects.toThrow( 'WebGPU renderer is required' );

			// train() resets `this._abortRequested = false` immediately after
			// validateTrainingSettings() succeeds, before the renderer check -
			// so even though this call ultimately rejects, the flag it reset
			// stays reset.
			expect( trainer._abortRequested ).toBe( false );

		} );

	} );

	describe( 'outputFeatures / opacityMode / alphaCutoff derivation (mocked GPU pipeline, no real renderer)', () => {

		function createFakeRenderer() {

			return {
				isWebGPURenderer: true,
				compute: vi.fn()
			};

		}

		function baseTrainerOptions() {

			return {
				iterations: 1,
				iblIterations: 0,
				batchSize: 4,
				hiddenSize: 4,
				iblHiddenSize: 4
			};

		}

		beforeEach( () => {

			createModel.mockClear();

		} );

		it( 'derives outputFeatures from teacher.supportsEmission/supportsOpacity, alphaCutoff from teacher.alphaCutoff, and opacityMode from teacher.opacityMode when no explicit opacityMode option is given', async () => {

			const renderer = createFakeRenderer();
			const teacher = {
				supportsEmission: true,
				supportsOpacity: true,
				alphaCutoff: 0.7,
				opacityMode: 'blend'
			};

			const trainer = new NeuralAppearanceTrainer( baseTrainerOptions() );
			await trainer.train( { material: {}, renderer, teacher } );

			expect( createModel ).toHaveBeenCalledTimes( 1 );
			const settings = createModel.mock.calls[ 0 ][ 0 ];

			expect( settings.outputFeatures ).toEqual( { emission: true, opacity: true } );
			expect( settings.alphaCutoff ).toBe( 0.7 );
			expect( settings.opacityMode ).toBe( 'blend' );

		} );

		it( 'derives outputFeatures as both false and falls back to alphaCutoff 0.5 / opacityMode "mask" when the teacher supports neither and reports no usable fields', async () => {

			const renderer = createFakeRenderer();
			const teacher = {
				supportsEmission: false,
				supportsOpacity: false,
				alphaCutoff: NaN,
				opacityMode: 'not-a-real-mode'
			};

			const trainer = new NeuralAppearanceTrainer( baseTrainerOptions() );
			await trainer.train( { material: {}, renderer, teacher } );

			const settings = createModel.mock.calls[ 0 ][ 0 ];

			expect( settings.outputFeatures ).toEqual( { emission: false, opacity: false } );
			expect( settings.alphaCutoff ).toBe( 0.5 );
			expect( settings.opacityMode ).toBe( 'mask' );

		} );

		it( 'an explicit options.opacityMode ("blend") overrides the teacher\'s own opacityMode ("mask")', async () => {

			const renderer = createFakeRenderer();
			const teacher = { supportsEmission: false, supportsOpacity: false, opacityMode: 'mask' };

			const trainer = new NeuralAppearanceTrainer( baseTrainerOptions() );
			await trainer.train( { material: {}, renderer, teacher, opacityMode: 'blend' } );

			const settings = createModel.mock.calls[ 0 ][ 0 ];
			expect( settings.opacityMode ).toBe( 'blend' );

		} );

		it( 'an explicit options.opacityMode that is not "mask"/"blend" is ignored, falling back to the teacher/default resolution instead', async () => {

			const renderer = createFakeRenderer();
			const teacher = { supportsEmission: false, supportsOpacity: false, opacityMode: 'blend' };

			const trainer = new NeuralAppearanceTrainer( baseTrainerOptions() );
			await trainer.train( { material: {}, renderer, teacher, opacityMode: 'bogus' } );

			const settings = createModel.mock.calls[ 0 ][ 0 ];
			// options.opacityMode ('bogus') isn't 'mask'/'blend', so it falls
			// through to the teacher's own (valid) opacityMode.
			expect( settings.opacityMode ).toBe( 'blend' );

		} );

	} );

} );

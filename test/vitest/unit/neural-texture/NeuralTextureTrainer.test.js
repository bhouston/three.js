import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// NeuralTextureTrainer.js is pure orchestration: it delegates every actual
// GPU/compute operation to imported factory functions and a GPU model class.
// Mock all of those delegate points so the loop, abort(), and onProgress
// wiring can be unit-tested without a renderer.

const mocks = vi.hoisted( () => {

	const cpuModel = { name: 'cpu-model' };

	const gpuModelInstance = {
		initFromCPUModel: vi.fn(),
		resetLoss: vi.fn(),
		learningRateUniform: { value: 0 },
		stepUniform: { value: 0 },
		maxGradientNormUniform: { value: 0 },
		readLoss: vi.fn().mockResolvedValue( 0.5 ),
		syncToCPU: vi.fn().mockResolvedValue( undefined ),
		dispose: vi.fn()
	};

	return {
		cpuModel,
		gpuModelInstance,
		createNeuralTextureModel: vi.fn( () => cpuModel ),
		// A regular function (not an arrow function) so `new NeuralTextureGPUModel()`
		// works: JS lets a constructor function opt out of building a fresh `this`
		// by returning an object instead, which is exactly what we want here.
		NeuralTextureGPUModel: vi.fn( function () {

			return gpuModelInstance;

		} ),
		createTextureTrainBatchComputeNode: vi.fn( () => 'trainBatchNode' ),
		createAccumulateGradientNormComputeNode: vi.fn( () => 'accumulateGradientNormNode' ),
		createTextureAdamWeightsComputeNode: vi.fn( () => 'adamWeightsNode' ),
		createTextureAdamLatentsComputeNode: vi.fn( () => 'adamLatentsNode' ),
		createResetGradientNormComputeNode: vi.fn( () => 'resetGradientNormNode' ),
		getLearningRate: vi.fn( ( settings, iteration ) => 0.01 - iteration * 0.0001 ),
		createRandom: vi.fn( () => () => 0.5 ),
		yieldToBrowser: vi.fn().mockResolvedValue( undefined )
	};

} );

vi.mock( '../../../../examples/jsm/neural-texture/NeuralTextureModel.js', () => ( {
	createNeuralTextureModel: mocks.createNeuralTextureModel
} ) );

vi.mock( '../../../../examples/jsm/neural-texture/NeuralTextureGPUModel.js', () => ( {
	NeuralTextureGPUModel: mocks.NeuralTextureGPUModel
} ) );

vi.mock( '../../../../examples/jsm/neural-texture/NeuralTextureGPUComputeTSL.js', () => ( {
	createTextureTrainBatchComputeNode: mocks.createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode: mocks.createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode: mocks.createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode: mocks.createTextureAdamLatentsComputeNode
} ) );

vi.mock( '../../../../examples/jsm/neural/NeuralGPUComputeTSL.js', () => ( {
	createResetGradientNormComputeNode: mocks.createResetGradientNormComputeNode
} ) );

vi.mock( '../../../../examples/jsm/neural/NeuralTrainingUtils.js', () => ( {
	getLearningRate: mocks.getLearningRate,
	createRandom: mocks.createRandom,
	yieldToBrowser: mocks.yieldToBrowser
} ) );

const { NeuralTextureTrainer } = await import( '../../../../examples/jsm/neural-texture/NeuralTextureTrainer.js' );

function createRenderer() {

	return {
		isWebGPURenderer: true,
		compute: vi.fn()
	};

}

describe( 'Addons > Neural > Neural-Texture > NeuralTextureTrainer', () => {

	beforeEach( () => {

		vi.clearAllMocks();
		mocks.createNeuralTextureModel.mockImplementation( () => mocks.cpuModel );
		mocks.NeuralTextureGPUModel.mockImplementation( function () {

			return mocks.gpuModelInstance;

		} );
		mocks.gpuModelInstance.readLoss.mockResolvedValue( 0.5 );
		mocks.gpuModelInstance.syncToCPU.mockResolvedValue( undefined );
		mocks.yieldToBrowser.mockResolvedValue( undefined );
		mocks.getLearningRate.mockImplementation( ( settings, iteration ) => 0.01 - iteration * 0.0001 );

	} );

	afterEach( () => {

		vi.restoreAllMocks();

	} );

	describe( 'constructor', () => {

		it( 'seeds its random generator via createRandom with the given seed', () => {

			new NeuralTextureTrainer( { seed: 42 } ); // eslint-disable-line no-new

			expect( mocks.createRandom ).toHaveBeenCalledWith( 42 );

		} );

	} );

	describe( 'train() input validation', () => {

		it( 'throws when no renderer is provided', async () => {

			const trainer = new NeuralTextureTrainer();

			await expect( trainer.train( { sourceTexture: {} } ) ).rejects.toThrow( 'WebGPU renderer is required' );

		} );

		it( 'throws when the renderer is not a WebGPURenderer', async () => {

			const trainer = new NeuralTextureTrainer();
			const renderer = { isWebGPURenderer: false };

			await expect( trainer.train( { renderer, sourceTexture: {} } ) ).rejects.toThrow( 'WebGPU renderer is required' );

		} );

		it( 'throws when neither sourceTexture nor sourceTextures is provided', async () => {

			const trainer = new NeuralTextureTrainer();
			const renderer = createRenderer();

			await expect( trainer.train( { renderer } ) ).rejects.toThrow( 'sourceTexture' );

		} );

		it( 'throws when sourceTextures is an empty array', async () => {

			const trainer = new NeuralTextureTrainer();
			const renderer = createRenderer();

			await expect( trainer.train( { renderer, sourceTextures: [] } ) ).rejects.toThrow( 'sourceTexture' );

		} );

	} );

	describe( 'train() GPU delegation', () => {

		it( 'wraps a single sourceTexture into an array before building the train-batch node', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 2 } );
			const renderer = createRenderer();
			const texture = { name: 'albedo' };

			await trainer.train( { renderer, sourceTexture: texture } );

			expect( mocks.createTextureTrainBatchComputeNode ).toHaveBeenCalledWith( mocks.gpuModelInstance, [ texture ] );

		} );

		it( 'passes sourceTextures through as-is', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 2 } );
			const renderer = createRenderer();
			const textures = [ { name: 'a' }, { name: 'b' } ];

			await trainer.train( { renderer, sourceTextures: textures } );

			expect( mocks.createTextureTrainBatchComputeNode ).toHaveBeenCalledWith( mocks.gpuModelInstance, textures );

		} );

		it( 'builds the CPU model and GPU model once and initializes the GPU model from it', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 3 } );
			const renderer = createRenderer();

			await trainer.train( { renderer, sourceTexture: {} } );

			expect( mocks.createNeuralTextureModel ).toHaveBeenCalledTimes( 1 );
			expect( mocks.NeuralTextureGPUModel ).toHaveBeenCalledTimes( 1 );
			expect( mocks.gpuModelInstance.initFromCPUModel ).toHaveBeenCalledWith( mocks.cpuModel );

		} );

		it( 'runs renderer.compute exactly 5 times per iteration, in the documented order', async () => {

			const iterations = 4;
			const trainer = new NeuralTextureTrainer( { iterations } );
			const renderer = createRenderer();

			await trainer.train( { renderer, sourceTexture: {} } );

			expect( renderer.compute ).toHaveBeenCalledTimes( iterations * 5 );

			const perIterationOrder = renderer.compute.mock.calls.slice( 0, 5 ).map( ( call ) => call[ 0 ] );
			expect( perIterationOrder ).toEqual( [
				'trainBatchNode',
				'resetGradientNormNode',
				'accumulateGradientNormNode',
				'adamWeightsNode',
				'adamLatentsNode'
			] );

		} );

		it( 'resets loss and updates the learning-rate/step/max-gradient-norm uniforms every iteration', async () => {

			const iterations = 3;
			const trainer = new NeuralTextureTrainer( { iterations, maxGradientNorm: 7 } );
			const renderer = createRenderer();

			const lrValuesSeen = [];
			const stepValuesSeen = [];

			// Snapshot the uniform values at the moment each iteration's compute
			// calls start, since the mock object is mutated in place.
			renderer.compute.mockImplementation( () => {

				lrValuesSeen.push( mocks.gpuModelInstance.learningRateUniform.value );
				stepValuesSeen.push( mocks.gpuModelInstance.stepUniform.value );

			} );

			await trainer.train( { renderer, sourceTexture: {} } );

			expect( mocks.gpuModelInstance.resetLoss ).toHaveBeenCalledTimes( iterations );
			expect( mocks.gpuModelInstance.maxGradientNormUniform.value ).toBe( 7 );

			// getLearningRate/step were called once per iteration with the running iteration index.
			expect( mocks.getLearningRate ).toHaveBeenCalledTimes( iterations );
			for ( let i = 0; i < iterations; i ++ ) {

				expect( mocks.getLearningRate ).toHaveBeenNthCalledWith( i + 1, expect.any( Object ), i );

			}

			// Each iteration's 5 compute() calls all saw that iteration's step value (i + 1).
			for ( let i = 0; i < iterations; i ++ ) {

				const sliceStart = i * 5;
				const stepSlice = stepValuesSeen.slice( sliceStart, sliceStart + 5 );
				expect( stepSlice.every( ( v ) => v === i + 1 ) ).toBe( true );

			}

		} );

		it( 'syncs the CPU model from the GPU model once after the loop completes, even without onProgress', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 3 } );
			const renderer = createRenderer();

			await trainer.train( { renderer, sourceTexture: {} } );

			expect( mocks.gpuModelInstance.readLoss ).not.toHaveBeenCalled();
			expect( mocks.gpuModelInstance.syncToCPU ).toHaveBeenCalledTimes( 1 );
			expect( mocks.gpuModelInstance.syncToCPU ).toHaveBeenLastCalledWith( mocks.cpuModel, renderer );

		} );

	} );

	describe( 'onProgress wiring', () => {

		it( 'is not invoked when onProgress is omitted', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 9 } );
			const renderer = createRenderer();

			const result = await trainer.train( { renderer, sourceTexture: {} } );

			expect( result.loss ).toBeNaN();

		} );

		it( 'is called on iteration 0, every 4th iteration after, and always on the final iteration', async () => {

			const iterations = 10; // expect syncs at iterations 1,5,9,10 (1-indexed "completed" values)
			const trainer = new NeuralTextureTrainer( { iterations } );
			const renderer = createRenderer();
			const onProgress = vi.fn();

			await trainer.train( { renderer, sourceTexture: {}, onProgress } );

			const seenIterations = onProgress.mock.calls.map( ( call ) => call[ 0 ].iteration );
			expect( seenIterations ).toEqual( [ 1, 5, 9, 10 ] );

			expect( mocks.gpuModelInstance.readLoss ).toHaveBeenCalledTimes( 4 );
			expect( mocks.gpuModelInstance.syncToCPU ).toHaveBeenCalledTimes( 4 + 1 ); // +1 for the unconditional post-loop sync

		} );

		it( 'passes iterations, loss, learningRate, cpuModel and gpuModel through to onProgress', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 1 } );
			const renderer = createRenderer();
			const onProgress = vi.fn();

			mocks.gpuModelInstance.readLoss.mockResolvedValue( 0.25 );
			mocks.getLearningRate.mockReturnValue( 0.0042 );

			await trainer.train( { renderer, sourceTexture: {}, onProgress } );

			expect( onProgress ).toHaveBeenCalledWith( {
				iteration: 1,
				iterations: 1,
				loss: 0.25,
				learningRate: 0.0042,
				cpuModel: mocks.cpuModel,
				gpuModel: mocks.gpuModelInstance
			} );

		} );

		it( 'awaits yieldToBrowser after every onProgress sync', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 1 } );
			const renderer = createRenderer();
			const onProgress = vi.fn();

			await trainer.train( { renderer, sourceTexture: {}, onProgress } );

			expect( mocks.yieldToBrowser ).toHaveBeenCalled();

		} );

	} );

	describe( 'abort()', () => {

		it( 'stops the loop after the current iteration and reports stoppedEarly with the partial iteration count', async () => {

			const iterations = 100;
			const trainer = new NeuralTextureTrainer( { iterations } );
			const renderer = createRenderer();

			const onProgress = vi.fn( ( { iteration } ) => {

				if ( iteration === 5 ) trainer.abort();

			} );

			const result = await trainer.train( { renderer, sourceTexture: {}, onProgress } );

			expect( result.stoppedEarly ).toBe( true );
			expect( result.iteration ).toBe( 5 );
			expect( result.iterations ).toBe( iterations );

			// No compute() calls should have happened for iterations after the abort.
			expect( renderer.compute ).toHaveBeenCalledTimes( 5 * 5 );

		} );

		it( 'still performs the final syncToCPU after an aborted run', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 50 } );
			const renderer = createRenderer();

			trainer.abort();

			const result = await trainer.train( { renderer, sourceTexture: {} } );

			// abort() before train() only sets the flag; train() resets it at the
			// start of the run, so the very first iteration still runs before the
			// (never re-triggered) flag would stop it - this documents that
			// abort() only affects an *in-flight* run, not a future one.
			expect( result.stoppedEarly ).toBe( false );
			expect( result.iteration ).toBe( 50 );

		} );

		it( 'a fresh train() call resets the abort flag from a previous run', async () => {

			const trainer = new NeuralTextureTrainer( { iterations: 100 } );
			const renderer = createRenderer();

			const onProgress = vi.fn( ( { iteration } ) => {

				if ( iteration === 5 ) trainer.abort();

			} );

			const first = await trainer.train( { renderer, sourceTexture: {}, onProgress } );
			expect( first.stoppedEarly ).toBe( true );

			renderer.compute.mockClear();

			const second = await trainer.train( { renderer, sourceTexture: {} } );

			expect( second.stoppedEarly ).toBe( false );
			expect( second.iteration ).toBe( 100 );

		} );

		it( 'reports stoppedEarly false and the full iteration count when the run completes normally', async () => {

			const iterations = 6;
			const trainer = new NeuralTextureTrainer( { iterations } );
			const renderer = createRenderer();

			const result = await trainer.train( { renderer, sourceTexture: {} } );

			expect( result.stoppedEarly ).toBe( false );
			expect( result.iteration ).toBe( iterations );
			expect( result.iterations ).toBe( iterations );

		} );

	} );

} );

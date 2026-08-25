import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createLatentGrid } from '../../../../examples/jsm/ntc/training/NTCGridModel.js';
import { createMLP, forwardMLP } from '../../../../examples/jsm/ntc/training/NTCMLP.js';
import { QUANTIZATION_SCHEMES, DEFAULT_QUANTIZATION_OPTIONS, computeLatentRanges } from '../../../../examples/jsm/ntc/training/NTCQuantization.js';

// Part 1: config plumbing (Task D) - mocks every GPU delegate point exactly
// like NTCTrainer.test.js, so `train()`'s orchestration (including
// the new QAT wiring: resolving `quantization`, forwarding it into
// `NTCGPUModel`, and freezing `trainer.quantizationRange` at the
// end) can be unit-tested without a renderer.
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
		dispose: vi.fn(),
		quantization: null,
		quantizationRangeUniforms: null,
		latentsBuffers: { attribute: 'latents-attribute' },
		layout: { gridLevels: [ { offset: 0, floatCount: 4 } ] },
		setQuantizationRange: vi.fn(),
		getQuantizationRange: vi.fn( () => [ [ - 1, 1 ] ] )
	};

	return {
		cpuModel,
		gpuModelInstance,
		createNTCGridPyramidModel: vi.fn( () => cpuModel ),
		NTCGPUModel: vi.fn( function () {

			return gpuModelInstance;

		} ),
		createTextureTrainBatchComputeNode: vi.fn( () => 'trainBatchNode' ),
		createAccumulateGradientNormComputeNode: vi.fn( () => 'accumulateGradientNormNode' ),
		createTextureAdamWeightsComputeNode: vi.fn( () => 'adamWeightsNode' ),
		createTextureAdamLatentsComputeNode: vi.fn( () => 'adamLatentsNode' ),
		createResetGradientNormComputeNode: vi.fn( () => 'resetGradientNormNode' ),
		getLearningRate: vi.fn( () => 0.01 ),
		createRandom: vi.fn( () => () => 0.5 ),
		yieldToBrowser: vi.fn().mockResolvedValue( undefined )
	};

} );

vi.mock( '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js', () => ( {
	createNTCGridPyramidModel: mocks.createNTCGridPyramidModel
} ) );

vi.mock( '../../../../examples/jsm/ntc/training/NTCGPUModel.js', () => ( {
	NTCGPUModel: mocks.NTCGPUModel
} ) );

vi.mock( '../../../../examples/jsm/ntc/training/NTCGPUComputeTSL.js', () => ( {
	createTextureTrainBatchComputeNode: mocks.createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode: mocks.createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode: mocks.createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode: mocks.createTextureAdamLatentsComputeNode
} ) );

vi.mock( '../../../../examples/jsm/ntc/training/NTCGPUKernelsTSL.js', () => ( {
	createResetGradientNormComputeNode: mocks.createResetGradientNormComputeNode
} ) );

vi.mock( '../../../../examples/jsm/ntc/training/NTCTrainingUtils.js', () => ( {
	getLearningRate: mocks.getLearningRate,
	createRandom: mocks.createRandom,
	yieldToBrowser: mocks.yieldToBrowser
} ) );

const { NTCTrainer } = await import( '../../../../examples/jsm/ntc/training/NTCTrainer.js' );

function createRenderer() {

	return {
		isWebGPURenderer: true,
		compute: vi.fn(),
		getArrayBufferAsync: vi.fn().mockResolvedValue( new Float32Array( 4 ).buffer )
	};

}

describe( 'Addons > Neural > Neural-Texture > NTCTrainer QAT config plumbing (Phase 2)', () => {

	beforeEach( () => {

		vi.clearAllMocks();
		mocks.createNTCGridPyramidModel.mockImplementation( () => mocks.cpuModel );
		mocks.NTCGPUModel.mockImplementation( function () {

			return mocks.gpuModelInstance;

		} );
		mocks.gpuModelInstance.readLoss.mockResolvedValue( 0.5 );
		mocks.gpuModelInstance.syncToCPU.mockResolvedValue( undefined );
		mocks.gpuModelInstance.getQuantizationRange.mockReturnValue( [ [ - 1, 1 ] ] );
		mocks.gpuModelInstance.quantization = null;
		mocks.yieldToBrowser.mockResolvedValue( undefined );
		mocks.getLearningRate.mockImplementation( () => 0.01 );

	} );

	afterEach( () => {

		vi.restoreAllMocks();

	} );

	it( 'DEFAULT_OPTIONS.quantization defaults to mode: "none" (a no-op)', () => {

		const trainer = new NTCTrainer();

		expect( trainer.options.quantization ).toEqual( DEFAULT_QUANTIZATION_OPTIONS );

	} );

	it( 'forwards a resolved quantization option through to NTCGPUModel', async () => {

		const trainer = new NTCTrainer( { iterations: 2, quantization: { mode: 'uint8', range: [ - 2, 2 ] } } );
		const renderer = createRenderer();

		await trainer.train( { renderer, sourceTexture: {} } );

		const constructedWith = mocks.NTCGPUModel.mock.calls[ 0 ][ 0 ];
		expect( constructedWith.quantization ).toEqual( { mode: 'uint8', range: [ - 2, 2 ] } );

	} );

	it( 'rejects an invalid quantization.mode before doing any GPU work', async () => {

		const trainer = new NTCTrainer( { iterations: 2, quantization: { mode: 'int4' } } );
		const renderer = createRenderer();

		await expect( trainer.train( { renderer, sourceTexture: {} } ) ).rejects.toThrow( /quantization\.mode/ );
		expect( mocks.NTCGPUModel ).not.toHaveBeenCalled();

	} );

	it( 'mode: "none" never reads back the GPU latent buffer or freezes a range', async () => {

		const trainer = new NTCTrainer( { iterations: 5 } );
		const renderer = createRenderer();

		const result = await trainer.train( { renderer, sourceTexture: {} } );

		expect( renderer.getArrayBufferAsync ).not.toHaveBeenCalledWith( mocks.gpuModelInstance.latentsBuffers.attribute );
		expect( result.quantizationRange ).toBeNull();
		expect( trainer.quantizationRange ).toBeNull();
		expect( mocks.cpuModel.quantizationRange ).toBeNull();

	} );

	it( 'mode: "uint8" freezes trainer.quantizationRange (and cpuModel.quantizationRange) from the GPU model at the end of training', async () => {

		mocks.gpuModelInstance.quantization = { mode: 'uint8', target: 'latents', range: 'auto', perLevel: true };
		mocks.gpuModelInstance.getQuantizationRange.mockReturnValue( [ [ - 0.4, 0.6 ] ] );

		const trainer = new NTCTrainer( { iterations: 3, quantization: { mode: 'uint8' } } );
		const renderer = createRenderer();

		const result = await trainer.train( { renderer, sourceTexture: {} } );

		expect( mocks.gpuModelInstance.getQuantizationRange ).toHaveBeenCalled();
		expect( trainer.quantizationRange ).toEqual( [ [ - 0.4, 0.6 ] ] );
		expect( result.quantizationRange ).toEqual( [ [ - 0.4, 0.6 ] ] );
		expect( mocks.cpuModel.quantizationRange ).toEqual( [ [ - 0.4, 0.6 ] ] );

	} );

} );

// Part 2: toy-fixture CPU regression test (Task E) demonstrating the actual
// forward-quantize mechanism NeuralTextureGPUComputeTSL.js's training kernel
// applies (see that file's QAT comment) - NeuralTextureModel.js itself has
// no CPU forward/backward pass to hook (training is 100% GPU-side, see this
// phase's final report), so this drives the same
// createLatentGrid/createMLP/forwardMLP/QUANTIZATION_SCHEMES primitives
// directly with a small hand-rolled numeric-gradient training loop.
function seededRandom( seed ) {

	let state = seed >>> 0;

	return function () {

		state = ( state + 0x6D2B79F5 ) | 0;
		let value = Math.imul( state ^ state >>> 15, 1 | state );
		value ^= value + Math.imul( value ^ value >>> 7, 61 | value );

		return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

	};

}

function bilinearSample( grid, uv, quantizeMode, range ) {

	const x = uv[ 0 ] * grid.width - 0.5;
	const y = uv[ 1 ] * grid.height - 0.5;
	const x0 = Math.floor( x );
	const y0 = Math.floor( y );
	const tx = x - x0;
	const ty = y - y0;
	const wrap = ( v, size ) => ( ( v % size ) + size ) % size;
	const taps = [
		{ x: wrap( x0, grid.width ), y: wrap( y0, grid.height ), weight: ( 1 - tx ) * ( 1 - ty ) },
		{ x: wrap( x0 + 1, grid.width ), y: wrap( y0, grid.height ), weight: tx * ( 1 - ty ) },
		{ x: wrap( x0, grid.width ), y: wrap( y0 + 1, grid.height ), weight: ( 1 - tx ) * ty },
		{ x: wrap( x0 + 1, grid.width ), y: wrap( y0 + 1, grid.height ), weight: tx * ty }
	];
	const output = new Array( grid.channels ).fill( 0 );

	for ( const tap of taps ) {

		const offset = ( tap.y * grid.width + tap.x ) * grid.channels;

		for ( let c = 0; c < grid.channels; c ++ ) output[ c ] += grid.data[ offset + c ] * tap.weight;

	}

	const scheme = QUANTIZATION_SCHEMES[ quantizeMode ];

	return output.map( ( v ) => scheme.quantizeForwardCPU( v, range[ 0 ], range[ 1 ] ) );

}

function collectParams( grid, mlp ) {

	const params = [ ...grid.data ];

	for ( const layer of mlp.layers ) params.push( ...layer.weights, ...layer.biases );

	return params;

}

function applyParams( grid, mlp, params ) {

	let cursor = 0;

	for ( let i = 0; i < grid.data.length; i ++ ) grid.data[ i ] = params[ cursor ++ ];

	for ( const layer of mlp.layers ) {

		for ( let i = 0; i < layer.weights.length; i ++ ) layer.weights[ i ] = params[ cursor ++ ];
		for ( let i = 0; i < layer.biases.length; i ++ ) layer.biases[ i ] = params[ cursor ++ ];

	}

}

/**
 * Batched finite-difference gradient descent over every parameter (grid
 * latents + decoder weights/biases, jointly - see collectParams/applyParams
 * and NeuralAppearanceModel.quantization.test.js's identical helper/doc
 * comment for why) using `bilinearSample`'s forward-quantize step.
 */
function trainToyTextureModel( { grid, mlp, uv, target, quantizeMode, range, iterations, learningRate } ) {

	const losses = [];

	const computeLoss = () => {

		const latents = bilinearSample( grid, uv, quantizeMode, range );
		const prediction = forwardMLP( mlp, latents ).output;
		let loss = 0;

		for ( let i = 0; i < target.length; i ++ ) {

			const diff = prediction[ i ] - target[ i ];
			loss += 0.5 * diff * diff;

		}

		return loss;

	};

	const epsilon = 1e-3;
	let params = collectParams( grid, mlp );

	for ( let iteration = 0; iteration < iterations; iteration ++ ) {

		applyParams( grid, mlp, params );
		const baseLoss = computeLoss();
		losses.push( baseLoss );

		const gradients = new Array( params.length );

		for ( let i = 0; i < params.length; i ++ ) {

			const original = params[ i ];
			params[ i ] = original + epsilon;
			applyParams( grid, mlp, params );
			const lossPlus = computeLoss();
			params[ i ] = original;

			gradients[ i ] = ( lossPlus - baseLoss ) / epsilon;

		}

		for ( let i = 0; i < params.length; i ++ ) params[ i ] -= learningRate * gradients[ i ];

	}

	applyParams( grid, mlp, params );
	losses.push( computeLoss() );

	return losses;

}

describe( 'Addons > Neural > Neural-Texture > toy QAT training loop (Phase 2)', () => {

	const uv = [ 0.35, 0.7 ];
	const target = [ 0.2, 0.5, - 0.1 ];
	const iterations = 18;
	const learningRate = 0.3;

	it( 'mode: "uint8" - loss decreases and final quantized latents land on <=256 discrete levels of the frozen range', () => {

		const random = seededRandom( 5 );
		const grid = createLatentGrid( 4, 4, 2, random );
		const mlp = createMLP( grid.channels, [ 6 ], target.length, random, 'relu', 'linear' );
		const range = [ - 1, 1 ];

		const losses = trainToyTextureModel( { grid, mlp, uv, target, quantizeMode: 'uint8', range, iterations, learningRate } );

		expect( losses[ losses.length - 1 ] ).toBeLessThan( losses[ 0 ] );

		// Freeze the final range from the trained latents (mirrors
		// NTCTrainer.js's end-of-training freeze) and confirm every
		// final quantized latent is exactly one of that range's 256 levels
		// (re-quantizing is a no-op - QUANTIZATION_SCHEMES.uint8 is idempotent,
		// see NeuralQuantization.test.js).
		const [ frozenRange ] = computeLatentRanges( grid.data, [ { offset: 0, floatCount: grid.data.length } ], true );
		const finalLatents = bilinearSample( grid, uv, 'uint8', frozenRange );

		for ( const value of finalLatents ) {

			const requantized = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( value, frozenRange[ 0 ], frozenRange[ 1 ] );
			expect( requantized ).toBeCloseTo( value, 9 );

		}

	} );

	it( 'mode: "none" - training still converges (QUANTIZATION_SCHEMES.none is a documented identity - see NeuralQuantization.test.js - so this is the same loop, unaffected)', () => {

		const random = seededRandom( 6 );
		const grid = createLatentGrid( 4, 4, 2, random );
		const mlp = createMLP( grid.channels, [ 6 ], target.length, random, 'relu', 'linear' );

		const losses = trainToyTextureModel( { grid, mlp, uv, target, quantizeMode: 'none', range: [ - 1, 1 ], iterations, learningRate } );

		expect( losses[ losses.length - 1 ] ).toBeLessThan( losses[ 0 ] );

	} );

} );

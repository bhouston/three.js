import { describe, expect, it } from 'vitest';
import { sampleLatents } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';
import { createLatentGrid } from '../../../../examples/jsm/neural/NeuralGridModel.js';
import { createMLP, forwardMLP } from '../../../../examples/jsm/neural/NeuralMLP.js';
import { QUANTIZATION_SCHEMES, computeLatentRanges } from '../../../../examples/jsm/neural/NeuralQuantization.js';

// Toy-fixture QAT regression test (Phase 2). NeuralAppearanceModel.sampleLatents
// is the one CPU forward-sampling path that exists in this codebase (see
// NeuralAppearanceModel.js's doc comment) - actual GPU training reuses the
// exact same QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL mirror (see
// NeuralAppearanceGPUComputeTSL.js/NeuralTextureGPUComputeTSL.js), so a hand-
// rolled CPU gradient-descent loop that (a) forward-quantizes sampled
// latents via `sampleLatents(grids, uv, quantization)` and (b) back-propagates
// straight through that quantize step (Straight-Through Estimator - see
// NeuralQuantization.js's module doc comment) exercises the same core
// mechanism the real trainers rely on, without needing a WebGPU renderer.

function makeToyGrid( random ) {

	// A single 4x4x2 grid level - small enough to be a fast toy fixture, big
	// enough (16 texels) to meaningfully land on <=256 uint8 levels.
	return createLatentGrid( 4, 4, 2, random );

}

function seededRandom( seed ) {

	let state = seed >>> 0;

	return function () {

		state = ( state + 0x6D2B79F5 ) | 0;
		let value = Math.imul( state ^ state >>> 15, 1 | state );
		value ^= value + Math.imul( value ^ value >>> 7, 61 | value );

		return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

	};

}

/**
 * Runs a tiny numeric-gradient-descent loop that fits `grid` + a small MLP
 * to a single fixed (uv, target) pair, using `quantization` (or `null`) as
 * the forward-quantize step `sampleLatents` applies - mirroring the GPU
 * trainers' Straight-Through Estimator: gradients are computed as if the
 * quantize step were identity (finite-difference gradients naturally do
 * this here, since they measure sensitivity of the *raw* latent to the
 * loss through whatever `sampleLatents` actually computes forward).
 */
/**
 * Flattens every trainable number this toy model has - the raw grid latents
 * *and* the decoder MLP's weights/biases, exactly like the real GPU trainers
 * jointly optimize both (see NeuralTextureGPUComputeTSL.js/
 * NeuralAppearanceGPUComputeTSL.js's Adam steps for weights and latents) -
 * into one flat array, so a single batched finite-difference gradient-
 * descent step can update all of them together each iteration.
 */
function collectParams( grid, mlp ) {

	const params = [ ...grid.data ];

	for ( const layer of mlp.layers ) {

		params.push( ...layer.weights, ...layer.biases );

	}

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
 * A tiny batched-gradient-descent loop over every parameter (grid latents +
 * decoder weights/biases - see `collectParams`/`applyParams`), using plain
 * central-difference-free (forward) finite differences for the gradient of
 * each parameter. `quantization` (or `null`) is the forward-quantize step
 * `sampleLatents` applies when reading the raw grid - since the gradient is
 * measured empirically (by perturbing the *raw* parameter and re-running the
 * full forward pass, quantize step included), this is exactly a Straight-
 * Through Estimator: whatever `quantizeForwardCPU`'s local slope actually is
 * gets baked into the finite-difference gradient at points away from a level
 * boundary, and rounds to zero exactly at one (mirroring - not equalling -
 * the real GPU trainers' *analytic* STE, which always treats the quantize
 * step as identity; see NeuralQuantization.js's module doc comment for why
 * that analytic simplification is valid here).
 */
function trainToyModel( { grid, mlp, uv, target, quantization, iterations, learningRate } ) {

	const losses = [];

	const computeLoss = () => {

		const latents = sampleLatents( [ grid ], uv, quantization ).output;
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

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceModel QAT (Phase 2)', () => {

	describe( 'sampleLatents quantization parameter', () => {

		it( 'defaults to no quantization (byte-for-byte identical to omitting the argument)', () => {

			const random = seededRandom( 1 );
			const grid = makeToyGrid( random );
			const uv = [ 0.37, 0.62 ];

			const withoutArg = sampleLatents( [ grid ], uv ).output;
			const withNullArg = sampleLatents( [ grid ], uv, null ).output;

			expect( withNullArg ).toEqual( withoutArg );

		} );

		it( '"none" mode reproduces the unquantized output exactly', () => {

			const random = seededRandom( 2 );
			const grid = makeToyGrid( random );
			const uv = [ 0.1, 0.9 ];

			const plain = sampleLatents( [ grid ], uv ).output;
			const quantizedNone = sampleLatents( [ grid ], uv, { mode: 'none', ranges: [ [ - 1, 1 ] ] } ).output;

			expect( quantizedNone ).toEqual( plain );

		} );

		it( '"uint8" mode snaps every level\'s sampled value onto one of 256 discrete levels within its range', () => {

			const random = seededRandom( 3 );
			const grid = makeToyGrid( random );
			const ranges = [ [ - 0.5, 0.5 ] ];
			const step = 1 / 255;

			for ( const uv of [ [ 0, 0 ], [ 0.25, 0.75 ], [ 0.6, 0.15 ], [ 0.99, 0.01 ] ] ) {

				const quantized = sampleLatents( [ grid ], uv, { mode: 'uint8', ranges } ).output;

				for ( const value of quantized ) {

					const clamped = Math.min( ranges[ 0 ][ 1 ], Math.max( ranges[ 0 ][ 0 ], value ) );
					const normalized = ( clamped - ranges[ 0 ][ 0 ] ) / ( ranges[ 0 ][ 1 ] - ranges[ 0 ][ 0 ] );
					const level = normalized / step;

					expect( level ).toBeCloseTo( Math.round( level ), 6 );

				}

			}

		} );

	} );

	describe( 'toy training loop', () => {

		const uv = [ 0.4, 0.65 ];
		const target = [ 0.3, - 0.2, 0.15 ];
		const iterations = 18;
		const learningRate = 0.3;

		it( 'mode: "uint8" - loss decreases across iterations and final quantized latents land on <=256 discrete levels', () => {

			const random = seededRandom( 11 );
			const grid = makeToyGrid( random );
			const mlp = createMLP( grid.channels, [ 6 ], target.length, random, 'relu', 'linear' );

			// A fixed, generous range - this test focuses on convergence +
			// discretization, not `range: 'auto'` tracking (that's exercised by
			// NeuralQuantization.test.js/the trainer-level range-tracking
			// behavior covered elsewhere in this phase).
			const ranges = [ [ - 1, 1 ] ];
			const quantization = { mode: 'uint8', ranges };

			const losses = trainToyModel( { grid, mlp, uv, target, quantization, iterations, learningRate } );

			expect( losses[ losses.length - 1 ] ).toBeLessThan( losses[ 0 ] );

			// The frozen range (see NeuralTextureTrainer.js/
			// NeuralAppearanceTrainer.js's `quantizationRange` freeze) would, in
			// a real run, be recomputed from the *final* trained latents - do
			// that here too, then verify every final quantized latent value is
			// exactly reproducible as one of the 256 uint8 levels of that
			// frozen range (i.e. re-quantizing is a no-op, matching
			// QUANTIZATION_SCHEMES.uint8's documented idempotency).
			const finalRanges = computeLatentRanges( grid.data, [ { offset: 0, floatCount: grid.data.length } ], true );
			const finalLatents = sampleLatents( [ grid ], uv, { mode: 'uint8', ranges: finalRanges } ).output;

			for ( const value of finalLatents ) {

				const requantized = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( value, finalRanges[ 0 ][ 0 ], finalRanges[ 0 ][ 1 ] );
				expect( requantized ).toBeCloseTo( value, 9 );

			}

		} );

		it( 'mode: "none" (or omitted) - identical loss trajectory either way, confirming QAT is opt-in only', () => {

			const iterationsShort = 6;

			const randomA = seededRandom( 21 );
			const gridA = makeToyGrid( randomA );
			const mlpA = createMLP( gridA.channels, [ 6 ], target.length, randomA, 'relu', 'linear' );
			const lossesNoQuantArg = trainToyModel( { grid: gridA, mlp: mlpA, uv, target, quantization: null, iterations: iterationsShort, learningRate } );

			const randomB = seededRandom( 21 );
			const gridB = makeToyGrid( randomB );
			const mlpB = createMLP( gridB.channels, [ 6 ], target.length, randomB, 'relu', 'linear' );
			const lossesModeNone = trainToyModel( { grid: gridB, mlp: mlpB, uv, target, quantization: { mode: 'none', ranges: [ [ - 1, 1 ] ] }, iterations: iterationsShort, learningRate } );

			expect( lossesModeNone ).toEqual( lossesNoQuantArg );
			expect( lossesNoQuantArg[ lossesNoQuantArg.length - 1 ] ).toBeLessThan( lossesNoQuantArg[ 0 ] );

		} );

	} );

} );

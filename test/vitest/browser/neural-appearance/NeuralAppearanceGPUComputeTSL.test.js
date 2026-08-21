import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { NeuralAppearanceGPUModel } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceGPUModel.js';
import {
	createTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createAdamWeightsComputeNode,
	createAdamLatentsComputeNode
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceGPUComputeTSL.js';
import { createResetGradientsComputeNode } from '../../../../examples/jsm/neural/NeuralGPUComputeTSL.js';
import { triangleWaveEncode } from '../../../../examples/jsm/neural/NeuralGridModel.js';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from '../../../../examples/jsm/neural/NeuralGPUTrainingConstants.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// NeuralAppearanceGPUComputeTSL.js is the biggest single file in the neural
// framework (training kernels: forward + cube-root-power-loss + backward for
// the full appearance decoder/IBL/indirect-probe heads, plus the shared Adam
// optimizer step). Rather than re-deriving the kernel's own arithmetic and
// asserting it matches itself (which would pass even if the kernel were
// wrong), every check below gets its expected value from somewhere
// independent of the kernel under test:
//
//  - the Adam optimizer tests hand-compute the update from the textbook
//    Adam formula (Kingma & Ba 2014), feeding the kernel a gradient we picked
//    ourselves (not one produced by the kernel's own backward pass);
//  - the gradient-norm test hand-sums squared gradients we wrote directly
//    into the fixed-point buffers;
//  - the "zero-weight sample" checks are a documented structural contract
//    (the `If (sampleWeight > 0)` guard) verified against the obvious
//    edge-case expectation - untouched buffers stay at their zero-initialized
//    value;
//  - the one weight-gradient cross-check for the actual forward/backward
//    training kernel uses central finite differencing of the loss - an
//    independent numerical method, not the kernel's own analytic formula -
//    to confirm the analytic gradient it writes into gradWeightsAtomic is
//    really the derivative of the loss it computes.
//
// Kept at the default levels=4 configuration throughout - see
// NeuralAppearanceModel.levels-bug.test.js / NeuralAppearanceTrainer.
// levels-bug.test.js for the known non-default-`levels` bug, which is out of
// scope here.

function mulberry32( seed ) {

	let a = seed >>> 0;
	return function () {

		a |= 0; a = ( a + 0x6D2B79F5 ) | 0;
		let t = Math.imul( a ^ ( a >>> 15 ), 1 | a );
		t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

function fillSmallRandom( array, seed, scale = 0.05 ) {

	const random = mulberry32( seed );
	for ( let i = 0; i < array.length; i ++ ) {

		array[ i ] = ( random() * 2 - 1 ) * scale;

	}

}

function normalize3( v ) {

	const len = Math.hypot( v[ 0 ], v[ 1 ], v[ 2 ] );
	return [ v[ 0 ] / len, v[ 1 ] / len, v[ 2 ] / len ];

}

// Standard Adam update (Kingma & Ba, 2014), worked out independently of the
// kernel under test: m_t = b1*m_{t-1} + (1-b1)*g ; v_t = b2*v_{t-1} +
// (1-b2)*g^2 ; bias-corrected mHat/vHat ; theta -= lr * mHat / (sqrt(vHat) + eps).
function referenceAdamStep( { weight, grad, m, v, lr, beta1, beta2, epsilon, step } ) {

	const nextM = beta1 * m + ( 1 - beta1 ) * grad;
	const nextV = beta2 * v + ( 1 - beta2 ) * grad * grad;
	const beta1Corr = Math.max( 1 - Math.pow( beta1, step ), 1e-10 );
	const beta2Corr = Math.max( 1 - Math.pow( beta2, step ), 1e-10 );
	const mHat = nextM / beta1Corr;
	const vHat = nextV / beta2Corr;
	const newWeight = weight - lr * mHat / ( Math.sqrt( Math.max( vHat, 0 ) ) + epsilon );

	return { newWeight, newM: nextM, newV: nextV };

}

async function readFloat32( renderer, attribute, index ) {

	const buffer = await renderer.getArrayBufferAsync( attribute );
	return new Float32Array( buffer )[ index ];

}

async function readInt32( renderer, attribute, index ) {

	const buffer = await renderer.getArrayBufferAsync( attribute );
	return new Int32Array( buffer )[ index ];

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceGPUComputeTSL (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	// A tiny model (small hiddenSize/base/target resolution) purely so
	// buffer allocation is cheap - the Adam/gradient-norm tests below write
	// directly into the fixed-point buffers rather than exercising the
	// forward/backward pass, so the model's numerical configuration doesn't
	// matter, only that it's a real NeuralAppearanceGPUModel with the real
	// buffer layout this kernel file expects.
	function createSmallModel( extra = {} ) {

		return new NeuralAppearanceGPUModel( {
			hiddenSize: 4,
			iblHiddenSize: 8,
			baseResolution: 4,
			growthFactor: 2,
			batchSize: 1,
			...extra
		} );

	}

	describe( 'createAdamWeightsComputeNode (Adam optimizer step)', () => {

		it( 'matches the hand-computed Adam update for one weight, step 1, no clipping', async () => {

			const model = createSmallModel();

			try {

				const index = 2;
				const w0 = 0.5, m0 = 0.1, v0 = 0.01, grad = 0.2;
				const lr = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7, step = 1;

				model.weightsBuffers.attribute.array[ index ] = w0;
				model.weightsBuffers.mAttribute.array[ index ] = m0;
				model.weightsBuffers.vAttribute.array[ index ] = v0;
				model.weightsBuffers.gradAttribute.array[ index ] = Math.round( grad * FIXED_POINT_SCALE );
				model.gradNormAttribute.array[ 0 ] = 0; // unclipped: clip scale resolves to 1
				model.weightsBuffers.attribute.needsUpdate = true;
				model.weightsBuffers.mAttribute.needsUpdate = true;
				model.weightsBuffers.vAttribute.needsUpdate = true;
				model.weightsBuffers.gradAttribute.needsUpdate = true;
				model.gradNormAttribute.needsUpdate = true;
				model.learningRateUniform.value = lr;
				model.stepUniform.value = step;
				model.maxGradientNormUniform.value = 1000;

				await renderer.computeAsync( createAdamWeightsComputeNode( model, { beta1, beta2, epsilon } ) );

				const expected = referenceAdamStep( { weight: w0, grad, m: m0, v: v0, lr, beta1, beta2, epsilon, step } );

				const gpuWeight = await readFloat32( renderer, model.weightsBuffers.attribute, index );
				const gpuM = await readFloat32( renderer, model.weightsBuffers.mAttribute, index );
				const gpuV = await readFloat32( renderer, model.weightsBuffers.vAttribute, index );

				expect( gpuWeight ).toBeCloseTo( expected.newWeight, 5 );
				expect( gpuM ).toBeCloseTo( expected.newM, 5 );
				expect( gpuV ).toBeCloseTo( expected.newV, 5 );

				// The gradient accumulator must be zeroed after being consumed,
				// so the next batch starts from a clean slate.
				const gpuGrad = await readInt32( renderer, model.weightsBuffers.gradAttribute, index );
				expect( gpuGrad ).toBe( 0 );

			} finally {

				model.dispose( renderer );

			}

		} );

		it( 'matches the hand-computed Adam update at step 2 (bias correction with nonzero prior moments)', async () => {

			const model = createSmallModel();

			try {

				const index = 5;
				const w0 = - 0.3, m0 = 0.05, v0 = 0.002, grad = - 0.15;
				const lr = 0.02, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7, step = 2;

				model.weightsBuffers.attribute.array[ index ] = w0;
				model.weightsBuffers.mAttribute.array[ index ] = m0;
				model.weightsBuffers.vAttribute.array[ index ] = v0;
				model.weightsBuffers.gradAttribute.array[ index ] = Math.round( grad * FIXED_POINT_SCALE );
				model.gradNormAttribute.array[ 0 ] = 0;
				model.weightsBuffers.attribute.needsUpdate = true;
				model.weightsBuffers.mAttribute.needsUpdate = true;
				model.weightsBuffers.vAttribute.needsUpdate = true;
				model.weightsBuffers.gradAttribute.needsUpdate = true;
				model.gradNormAttribute.needsUpdate = true;
				model.learningRateUniform.value = lr;
				model.stepUniform.value = step;
				model.maxGradientNormUniform.value = 1000;

				await renderer.computeAsync( createAdamWeightsComputeNode( model, { beta1, beta2, epsilon } ) );

				const expected = referenceAdamStep( { weight: w0, grad, m: m0, v: v0, lr, beta1, beta2, epsilon, step } );

				const gpuWeight = await readFloat32( renderer, model.weightsBuffers.attribute, index );
				const gpuM = await readFloat32( renderer, model.weightsBuffers.mAttribute, index );
				const gpuV = await readFloat32( renderer, model.weightsBuffers.vAttribute, index );

				expect( gpuWeight ).toBeCloseTo( expected.newWeight, 5 );
				expect( gpuM ).toBeCloseTo( expected.newM, 5 );
				expect( gpuV ).toBeCloseTo( expected.newV, 5 );

			} finally {

				model.dispose( renderer );

			}

		} );

		it( 'applies the global gradient-norm clip scale before the Adam update', async () => {

			const model = createSmallModel();

			try {

				const index = 1;
				const w0 = 0.0, m0 = 0.0, v0 = 0.0, rawGrad = 10.0;
				const lr = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7, step = 1;
				const maxGradientNorm = 1.0;

				// gradNormAtomic represents sum(grad^2) in fixed point; a lone
				// gradient of magnitude 10 gives normSquared = 100, so
				// ||grad|| = 10 and clipScale = maxGradientNorm / ||grad|| = 0.1.
				const normSquared = rawGrad * rawGrad;
				const clipScale = Math.min( 1, maxGradientNorm / Math.sqrt( normSquared ) );
				const clippedGrad = rawGrad * clipScale;

				model.weightsBuffers.attribute.array[ index ] = w0;
				model.weightsBuffers.mAttribute.array[ index ] = m0;
				model.weightsBuffers.vAttribute.array[ index ] = v0;
				model.weightsBuffers.gradAttribute.array[ index ] = Math.round( rawGrad * FIXED_POINT_SCALE );
				model.gradNormAttribute.array[ 0 ] = Math.round( normSquared * GRADIENT_NORM_SCALE );
				model.weightsBuffers.attribute.needsUpdate = true;
				model.weightsBuffers.mAttribute.needsUpdate = true;
				model.weightsBuffers.vAttribute.needsUpdate = true;
				model.weightsBuffers.gradAttribute.needsUpdate = true;
				model.gradNormAttribute.needsUpdate = true;
				model.learningRateUniform.value = lr;
				model.stepUniform.value = step;
				model.maxGradientNormUniform.value = maxGradientNorm;

				await renderer.computeAsync( createAdamWeightsComputeNode( model, { beta1, beta2, epsilon } ) );

				const expectedClipped = referenceAdamStep( { weight: w0, grad: clippedGrad, m: m0, v: v0, lr, beta1, beta2, epsilon, step } );
				const expectedUnclipped = referenceAdamStep( { weight: w0, grad: rawGrad, m: m0, v: v0, lr, beta1, beta2, epsilon, step } );

				const gpuWeight = await readFloat32( renderer, model.weightsBuffers.attribute, index );
				const gpuM = await readFloat32( renderer, model.weightsBuffers.mAttribute, index );

				expect( gpuWeight ).toBeCloseTo( expectedClipped.newWeight, 5 );
				expect( gpuM ).toBeCloseTo( expectedClipped.newM, 5 );
				// Sanity that clipping actually took effect: with zero prior
				// moments, Adam's step-1 weight update is nearly scale-invariant
				// to the gradient's magnitude (mHat/sqrt(vHat) ~= sign(grad)),
				// so the *weight* alone barely differs whether the raw gradient
				// was 10 or the clipped 1 - that's expected Adam behavior, not a
				// sign the clip was skipped. The first moment `m` is where the
				// 10x-vs-1x gradient scale actually shows up untouched by that
				// normalization, so that's what proves clipping ran.
				expect( Math.abs( expectedClipped.newM - expectedUnclipped.newM ) ).toBeGreaterThan( 0.5 );

			} finally {

				model.dispose( renderer );

			}

		} );

	} );

	describe( 'createAdamLatentsComputeNode (Adam optimizer step, latent grid buffer)', () => {

		it( 'matches the hand-computed Adam update for one latent value', async () => {

			const model = createSmallModel();

			try {

				const index = 3;
				const z0 = 0.02, m0 = - 0.01, v0 = 0.0005, grad = 0.08;
				const lr = 0.005, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7, step = 1;

				model.latentsBuffers.attribute.array[ index ] = z0;
				model.latentsBuffers.mAttribute.array[ index ] = m0;
				model.latentsBuffers.vAttribute.array[ index ] = v0;
				model.latentsBuffers.gradAttribute.array[ index ] = Math.round( grad * FIXED_POINT_SCALE );
				model.gradNormAttribute.array[ 0 ] = 0;
				model.latentsBuffers.attribute.needsUpdate = true;
				model.latentsBuffers.mAttribute.needsUpdate = true;
				model.latentsBuffers.vAttribute.needsUpdate = true;
				model.latentsBuffers.gradAttribute.needsUpdate = true;
				model.gradNormAttribute.needsUpdate = true;
				model.learningRateUniform.value = lr;
				model.stepUniform.value = step;
				model.maxGradientNormUniform.value = 1000;

				await renderer.computeAsync( createAdamLatentsComputeNode( model, { beta1, beta2, epsilon } ) );

				const expected = referenceAdamStep( { weight: z0, grad, m: m0, v: v0, lr, beta1, beta2, epsilon, step } );

				const gpuValue = await readFloat32( renderer, model.latentsBuffers.attribute, index );
				const gpuM = await readFloat32( renderer, model.latentsBuffers.mAttribute, index );
				const gpuV = await readFloat32( renderer, model.latentsBuffers.vAttribute, index );

				expect( gpuValue ).toBeCloseTo( expected.newWeight, 5 );
				expect( gpuM ).toBeCloseTo( expected.newM, 5 );
				expect( gpuV ).toBeCloseTo( expected.newV, 5 );

			} finally {

				model.dispose( renderer );

			}

		} );

	} );

	describe( 'createAccumulateGradientNormComputeNode', () => {

		it( 'sums squared gradients across weights and latents, matching a hand-computed sum', async () => {

			const model = createSmallModel();

			try {

				const gW1 = 0.3, gW2 = - 0.4, gL1 = 0.1, gL2 = 0.2;

				model.weightsBuffers.gradAttribute.array[ 0 ] = Math.round( gW1 * FIXED_POINT_SCALE );
				model.weightsBuffers.gradAttribute.array[ 1 ] = Math.round( gW2 * FIXED_POINT_SCALE );
				model.latentsBuffers.gradAttribute.array[ 0 ] = Math.round( gL1 * FIXED_POINT_SCALE );
				model.latentsBuffers.gradAttribute.array[ 1 ] = Math.round( gL2 * FIXED_POINT_SCALE );
				model.gradNormAttribute.array[ 0 ] = 0;
				model.weightsBuffers.gradAttribute.needsUpdate = true;
				model.latentsBuffers.gradAttribute.needsUpdate = true;
				model.gradNormAttribute.needsUpdate = true;

				await renderer.computeAsync( createAccumulateGradientNormComputeNode( model ) );

				const expectedSum = gW1 * gW1 + gW2 * gW2 + gL1 * gL1 + gL2 * gL2;
				const expectedFixedPoint = Math.round( expectedSum * GRADIENT_NORM_SCALE );

				const gpuValue = await readInt32( renderer, model.gradNormAttribute, 0 );

				// Every weight/latent beyond the four set above is exactly 0
				// (fresh Int32Array), so they contribute nothing - the sum
				// should match to within per-element fixed-point rounding.
				expect( gpuValue ).toBeGreaterThanOrEqual( expectedFixedPoint - 4 );
				expect( gpuValue ).toBeLessThanOrEqual( expectedFixedPoint + 4 );

			} finally {

				model.dispose( renderer );

			}

		} );

		it( 'excludes latents when includeLatents is false', async () => {

			const model = createSmallModel();

			try {

				const gW1 = 0.5, gL1 = 100.0; // latent gradient deliberately huge - must not leak in

				model.weightsBuffers.gradAttribute.array[ 0 ] = Math.round( gW1 * FIXED_POINT_SCALE );
				model.latentsBuffers.gradAttribute.array[ 0 ] = Math.round( gL1 * FIXED_POINT_SCALE );
				model.gradNormAttribute.array[ 0 ] = 0;
				model.weightsBuffers.gradAttribute.needsUpdate = true;
				model.latentsBuffers.gradAttribute.needsUpdate = true;
				model.gradNormAttribute.needsUpdate = true;

				await renderer.computeAsync( createAccumulateGradientNormComputeNode( model, { includeLatents: false } ) );

				const expectedFixedPoint = Math.round( gW1 * gW1 * GRADIENT_NORM_SCALE );
				const gpuValue = await readInt32( renderer, model.gradNormAttribute, 0 );

				expect( gpuValue ).toBeGreaterThanOrEqual( expectedFixedPoint - 4 );
				expect( gpuValue ).toBeLessThanOrEqual( expectedFixedPoint + 4 );

			} finally {

				model.dispose( renderer );

			}

		} );

	} );

	describe( 'createTrainBatchComputeNode', () => {

		function createTrainModel() {

			return new NeuralAppearanceGPUModel( {
				hiddenSize: 8,
				iblHiddenSize: 8,
				baseResolution: 4,
				growthFactor: 2,
				batchSize: 1
			} );

		}

		function seedModel( model, seed ) {

			fillSmallRandom( model.weightsBuffers.attribute.array, seed );
			fillSmallRandom( model.latentsBuffers.attribute.array, seed + 1 );
			model.weightsBuffers.attribute.needsUpdate = true;
			model.latentsBuffers.attribute.needsUpdate = true;

		}

		async function resetAndRun( model ) {

			await renderer.computeAsync( createResetGradientsComputeNode( model ) );
			model.resetLoss();
			await renderer.computeAsync( createTrainBatchComputeNode( model ) );

		}

		it( 'a zero-weight sample contributes nothing - loss and its activation slot stay at their zero-initialized value (structural guard)', async () => {

			const model = createTrainModel();

			try {

				seedModel( model, 12345 );
				model.uploadSamples( [ {
					uv: [ 0.4, 0.6 ],
					wi: [ 0, 0, 1 ],
					wo: [ 0, 0, 1 ],
					target: [ 0.3, 0.4, 0.5 ],
					weight: 0, // <- guard: If (sampleWeight > 0) never enters
					iblWeight: 0
				} ] );

				await resetAndRun( model );

				const losses = await model.readLosses( renderer );
				expect( losses.loss ).toBe( 0 );
				expect( losses.directLoss ).toBe( 0 );
				expect( losses.iblLoss ).toBe( 0 );

				// actA0Offset (index 0 of the per-sample activation block) is
				// never written for a skipped sample - it must still read back
				// as the buffer's zero-initialized default, not some leftover
				// or garbage value.
				const actA0 = await readFloat32( renderer, model.activationsAttribute, model.layout.actA0Offset );
				expect( actA0 ).toBe( 0 );

			} finally {

				model.dispose( renderer );

			}

		} );

		it( 'the analytic weight gradient from the backward pass matches central finite-differencing of the loss (independent numerical cross-check)', async () => {

			const model = createTrainModel();

			try {

				seedModel( model, 777 );

				const baseWeights = Float32Array.from( model.weightsBuffers.attribute.array );
				// Row j=1 (green channel), input i=0, of the final linear (RGB)
				// layer. Row j=0 lands in the output clamp's "dead zone"
				// (z3 < 0) for this seed, where the kernel intentionally uses a
				// leaky (non-differentiable-matching) clamp gradient instead of
				// the true local derivative - see OUTPUT_CLAMP_GRADIENT_LEAK in
				// the source - so finite-differencing it wouldn't cross-check
				// anything. Row j=1's z3 is comfortably positive (checked via
				// actZ3Offset below), so its loss is smooth there and the
				// analytic gradient really is the local derivative.
				const targetIndex = model.layout.layer2WeightsOffset + model.layout.hiddenSize;

				const sample = {
					uv: [ 0.37, 0.61 ],
					wi: normalize3( [ 0.2, 0.1, 0.95 ] ),
					wo: normalize3( [ - 0.1, 0.3, 0.9 ] ),
					target: [ 0.15, 0.35, 0.55 ],
					weight: 1,
					iblWeight: 0
				};
				model.uploadSamples( [ sample ] );

				const epsilon = 1e-3;

				async function lossForWeight( value ) {

					model.weightsBuffers.attribute.array.set( baseWeights );
					model.weightsBuffers.attribute.array[ targetIndex ] = value;
					model.weightsBuffers.attribute.needsUpdate = true;
					await resetAndRun( model );
					const losses = await model.readLosses( renderer );
					return losses.loss;

				}

				const base = baseWeights[ targetIndex ];
				const lossPlus = await lossForWeight( base + epsilon );
				const lossMinus = await lossForWeight( base - epsilon );
				const centralDifference = ( lossPlus - lossMinus ) / ( 2 * epsilon );

				// Analytic gradient: restore the exact base weight and read what
				// the kernel's own backward pass wrote into gradWeightsAtomic.
				model.weightsBuffers.attribute.array.set( baseWeights );
				model.weightsBuffers.attribute.needsUpdate = true;
				await resetAndRun( model );

				const rawGrad = await readInt32( renderer, model.weightsBuffers.gradAttribute, targetIndex );
				const analyticGradient = rawGrad / FIXED_POINT_SCALE;

				// Confirm the precondition documented above: channel j=1's z3
				// really is in the smooth (un-clamped) region for this seed, so
				// this comparison is meaningful rather than accidentally
				// landing in the leaky-clamp dead zone.
				const activationsBuffer = await renderer.getArrayBufferAsync( model.activationsAttribute );
				const z3Green = new Float32Array( activationsBuffer )[ model.layout.actZ3Offset + 1 ];
				expect( z3Green ).toBeGreaterThan( 0 );

				// Sanity: make sure this isn't a trivial 0 ~= 0 comparison.
				expect( Math.abs( analyticGradient ) ).toBeGreaterThan( 1e-4 );

				expect( analyticGradient ).toBeCloseTo( centralDifference, 2 );

			} finally {

				model.dispose( renderer );

			}

		} );

		it( 'with peOctaves > 0, the analytic gradient of a layer0 weight reading a positional-encoding input still matches central finite-differencing', async () => {

			// Targets a decoder layer0 weight whose *column* (input index) falls
			// inside the PE block, not the grid-latent or frame-dot columns - a
			// generic-but-wrong layer0 backward pass (e.g. one that silently
			// stopped at the old, narrower decoderInputSize) would either throw
			// on an out-of-bounds column or produce a gradient unrelated to this
			// weight's actual local derivative, and this test would catch either.
			const model = new NeuralAppearanceGPUModel( {
				hiddenSize: 8,
				iblHiddenSize: 8,
				baseResolution: 4,
				growthFactor: 2,
				peOctaves: 2,
				inputEncoding: 'positional',
				batchSize: 1
			} );

			try {

				fillSmallRandom( model.weightsBuffers.attribute.array, 999 );
				fillSmallRandom( model.latentsBuffers.attribute.array, 1000 );
				model.weightsBuffers.attribute.needsUpdate = true;
				model.latentsBuffers.attribute.needsUpdate = true;

				const baseWeights = Float32Array.from( model.weightsBuffers.attribute.array );
				const { layer0WeightsOffset, decoderInputSize, decoderPeOffset } = model.layout;
				const outputRow = 0;
				const peColumn = decoderPeOffset; // first PE input column
				const targetIndex = layer0WeightsOffset + outputRow * decoderInputSize + peColumn;

				const sample = {
					uv: [ 0.37, 0.61 ],
					wi: normalize3( [ 0.2, 0.1, 0.95 ] ),
					wo: normalize3( [ - 0.1, 0.3, 0.9 ] ),
					target: [ 0.15, 0.35, 0.55 ],
					weight: 1,
					iblWeight: 0
				};
				model.uploadSamples( [ sample ] );

				// A much smaller epsilon than the layer2 (output-adjacent) test
				// above uses: this weight is 2 layers further from the loss, so
				// its perturbation is amplified through 2 extra ReLU
				// nonlinearities before reaching the loss - a larger epsilon risks
				// a downstream unit crossing its ReLU kink between the +/-epsilon
				// probes, which would corrupt the finite-difference estimate
				// itself rather than reveal anything about the analytic gradient.
				const epsilon = 1e-5;

				async function lossForWeight( value ) {

					model.weightsBuffers.attribute.array.set( baseWeights );
					model.weightsBuffers.attribute.array[ targetIndex ] = value;
					model.weightsBuffers.attribute.needsUpdate = true;
					await resetAndRun( model );
					const losses = await model.readLosses( renderer );
					return losses.loss;

				}

				const base = baseWeights[ targetIndex ];
				const lossPlus = await lossForWeight( base + epsilon );
				const lossMinus = await lossForWeight( base - epsilon );
				const centralDifference = ( lossPlus - lossMinus ) / ( 2 * epsilon );

				model.weightsBuffers.attribute.array.set( baseWeights );
				model.weightsBuffers.attribute.needsUpdate = true;
				await resetAndRun( model );

				const rawGrad = await readInt32( renderer, model.weightsBuffers.gradAttribute, targetIndex );
				const analyticGradient = rawGrad / FIXED_POINT_SCALE;

				// Confirm this hidden unit's pre-activation is comfortably clear of
				// the ReLU kink at z1=0 - a target this close to the kink would
				// make finite-differencing itself unreliable (the +/-epsilon probes
				// could straddle the kink and see a different local slope than the
				// analytic gradient at the unperturbed weight), independent of
				// whether the kernel's own gradient is correct.
				const activationsBuffer = await renderer.getArrayBufferAsync( model.activationsAttribute );
				const z1Row0 = new Float32Array( activationsBuffer )[ model.layout.actZ1Offset + outputRow ];
				expect( Math.abs( z1Row0 ) ).toBeGreaterThan( 100 * epsilon );

				expect( Math.abs( analyticGradient ) ).toBeGreaterThan( 1e-5 );
				expect( analyticGradient ).toBeCloseTo( centralDifference, 1 );

			} finally {

				model.dispose( renderer );

			}

		} );

		it( 'writes the NTC-style tiled positional encoding (peOctaves > 0) into a0/iblA0/indirectA0 at the offsets NeuralAppearanceGPUModel computed, matching the independent CPU reference (NeuralGridModel.triangleWaveEncode)', async () => {

			// iblWeight: 1 below exercises the IBL + both indirect-probe blocks
			// too (unconditionally allocated - unlike emission/opacity, there's
			// no supportsX flag gating them), so this one sample checks all 3
			// input vectors' PE copies in a single kernel run.
			const model = new NeuralAppearanceGPUModel( {
				hiddenSize: 8,
				iblHiddenSize: 8,
				baseResolution: 4,
				growthFactor: 2,
				peOctaves: 2,
				inputEncoding: 'positional',
				batchSize: 1
			} );

			try {

				fillSmallRandom( model.weightsBuffers.attribute.array, 555 );
				fillSmallRandom( model.latentsBuffers.attribute.array, 556 );
				model.weightsBuffers.attribute.needsUpdate = true;
				model.latentsBuffers.attribute.needsUpdate = true;

				const uv = [ 0.37, 0.61 ];
				model.uploadSamples( [ {
					uv,
					wi: normalize3( [ 0.2, 0.1, 0.95 ] ),
					wo: normalize3( [ - 0.1, 0.3, 0.9 ] ),
					target: [ 0.15, 0.35, 0.55 ],
					weight: 1,
					iblWeight: 1,
					iblDirection: [ 0, 0, 1 ],
					iblRoughness: 0.5
				} ] );

				await resetAndRun( model );

				const expectedPe = triangleWaveEncode( uv[ 0 ], uv[ 1 ], model.layout.peOctaves );
				expect( expectedPe.length ).toBe( 4 ); // peOctaves=2 -> 2*2

				const activationsBuffer = await renderer.getArrayBufferAsync( model.activationsAttribute );
				const activations = new Float32Array( activationsBuffer );

				for ( let i = 0; i < expectedPe.length; i ++ ) {

					const a0Value = activations[ model.layout.actA0Offset + model.layout.decoderPeOffset + i ];
					expect( a0Value, `a0 PE[${ i }]` ).toBeCloseTo( expectedPe[ i ], 5 );

					const iblA0Value = activations[ model.layout.actIblA0Offset + model.layout.iblPeOffset + i ];
					expect( iblA0Value, `iblA0 PE[${ i }]` ).toBeCloseTo( expectedPe[ i ], 5 );

					const indirectA0Value = activations[ model.layout.actIndirectA0Offset + model.layout.indirectPeOffset + i ];
					expect( indirectA0Value, `indirectA0 PE[${ i }]` ).toBeCloseTo( expectedPe[ i ], 5 );

				}

			} finally {

				model.dispose( renderer );

			}

		} );

	} );

} );

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
	createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode
} from '../../../../examples/jsm/neural-texture/NeuralTextureGPUComputeTSL.js';
import { NeuralTextureGPUModel } from '../../../../examples/jsm/neural-texture/NeuralTextureGPUModel.js';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from '../../../../examples/jsm/neural/NeuralGPUTrainingConstants.js';
import { withTestRenderer } from '../helpers/webgpuEval.js';

// NeuralTextureGPUComputeTSL.js is the actual training-step compute kernel
// file for the neural-texture trainer: forward MLP + L2 loss + hand-written
// backward pass + Adam optimizer step, all running as real WGSL compute
// shaders reading/writing fixed-point atomic storage buffers (see
// NeuralGPUComputeTSL.js's createAdamComputeNode and
// NeuralGPUTrainingConstants.js for the shared fixed-point convention).
//
// Every reference value below is derived independently from the published
// Adam optimizer update rule (Kingma & Ba, 2015) and from first-principles
// calculus for the single-weight linear-regression backward pass - not by
// re-reading the kernel's own arithmetic and asserting it matches itself.
// A genuine regression (wrong bias-correction exponent, swapped beta1/beta2,
// a sign error in the backward pass, gradient clipping applied twice, etc.)
// would make these fail.

/**
 * Independent, hand-written reference implementation of one Adam step,
 * transcribed directly from the Kingma & Ba (2015) update rule:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *   m_hat = m_t / (1 - beta1^t)
 *   v_hat = v_t / (1 - beta2^t)
 *   theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
 */
function referenceAdamStep( { weight, grad, m, v, step, lr, beta1, beta2, epsilon } ) {

	const nextM = beta1 * m + ( 1 - beta1 ) * grad;
	const nextV = beta2 * v + ( 1 - beta2 ) * grad * grad;
	const mHat = nextM / ( 1 - Math.pow( beta1, step ) );
	const vHat = nextV / ( 1 - Math.pow( beta2, step ) );
	const nextWeight = weight - lr * mHat / ( Math.sqrt( vHat ) + epsilon );

	return { weight: nextWeight, m: nextM, v: nextV };

}

// Fixed-point quantization (see NeuralGPUTrainingConstants.js) truncates
// toward zero before every atomic accumulation, so a single-sample gradient
// deposited at exactly FIXED_POINT_SCALE resolution can be off by up to one
// quantum (1e-5) from the value it was meant to represent.
const QUANTUM = 1 / FIXED_POINT_SCALE;

function readFloat( renderer, attribute, index ) {

	return renderer.getArrayBufferAsync( attribute ).then( ( buffer ) => new Float32Array( buffer )[ index ] );

}

function readInt( renderer, attribute, index ) {

	return renderer.getArrayBufferAsync( attribute ).then( ( buffer ) => new Int32Array( buffer )[ index ] );

}

// Builds a tiny NeuralTextureGPUModel: 1 grid channel, 1 level (2x2), no
// hidden layers, 1 output channel - just enough buffer layout (2 weights:
// one input weight + one bias, 4 latents) to address individual elements by
// hand while exercising the exact same kernels the real trainer runs.
function createTinyGPUModel() {

	return new NeuralTextureGPUModel( {
		channels: 1,
		levels: 1,
		baseResolution: 2,
		growthFactor: 2,
		hiddenSizes: [],
		outputChannels: 1,
		batchSize: 1
	} );

}

describe( 'Addons > NeuralTexture > NeuralTextureGPUComputeTSL (real WebGPU)', () => {

	const getRenderer = withTestRenderer( { beforeAll, afterAll } );

	describe( 'createTextureAdamWeightsComputeNode (Adam optimizer step)', () => {

		it( 'matches the hand-derived Kingma & Ba Adam update for a single weight', async () => {

			const renderer = getRenderer();
			const gpuModel = createTinyGPUModel();

			const weight = 0.4;
			const grad = 0.125;
			const m = 0.02;
			const v = 0.0009;
			const step = 5;
			const lr = 0.01;
			const beta1 = 0.9;
			const beta2 = 0.999;
			const epsilon = 1e-7;

			const idx = 0;
			gpuModel.weightsBuffers.attribute.array[ idx ] = weight;
			gpuModel.weightsBuffers.attribute.needsUpdate = true;
			gpuModel.weightsBuffers.gradAttribute.array[ idx ] = Math.round( grad * FIXED_POINT_SCALE );
			gpuModel.weightsBuffers.gradAttribute.needsUpdate = true;
			gpuModel.weightsBuffers.mAttribute.array[ idx ] = m;
			gpuModel.weightsBuffers.mAttribute.needsUpdate = true;
			gpuModel.weightsBuffers.vAttribute.array[ idx ] = v;
			gpuModel.weightsBuffers.vAttribute.needsUpdate = true;
			gpuModel.gradNormAttribute.array[ 0 ] = 0; // norm-squared 0 -> clip scale clamps to 1 (no clipping)
			gpuModel.gradNormAttribute.needsUpdate = true;
			gpuModel.stepUniform.value = step;
			gpuModel.learningRateUniform.value = lr;
			gpuModel.invBatchUniform.value = 1; // batchSize is 1 here, so no averaging effect
			gpuModel.maxGradientNormUniform.value = 1000; // large enough that clipping never engages

			const kernel = createTextureAdamWeightsComputeNode( gpuModel, { beta1, beta2, epsilon } );
			await renderer.computeAsync( kernel );

			const [ gpuWeight, gpuM, gpuV ] = await Promise.all( [
				readFloat( renderer, gpuModel.weightsBuffers.attribute, idx ),
				readFloat( renderer, gpuModel.weightsBuffers.mAttribute, idx ),
				readFloat( renderer, gpuModel.weightsBuffers.vAttribute, idx )
			] );

			const expected = referenceAdamStep( { weight, grad, m, v, step, lr, beta1, beta2, epsilon } );

			expect( gpuWeight ).toBeCloseTo( expected.weight, 4 );
			expect( gpuM ).toBeCloseTo( expected.m, 4 );
			expect( gpuV ).toBeCloseTo( expected.v, 6 );

			// The gradient accumulator must be zeroed after being consumed, so the
			// next training iteration starts from a clean accumulator.
			const gradAfter = await readInt( renderer, gpuModel.weightsBuffers.gradAttribute, idx );
			expect( gradAfter ).toBe( 0 );

		} );

		it( 'applies global gradient-norm clipping (clip-by-global-norm) before the Adam update', async () => {

			const renderer = getRenderer();
			const gpuModel = createTinyGPUModel();

			const weight = 1.0;
			const rawGrad = 2.0; // large gradient
			// Nonzero prior moment estimates and step > 1 matter here: at step 1
			// with m = v = 0, Adam's own v-normalization makes the update
			// magnitude ~lr*sign(grad) regardless of grad's magnitude, which
			// would mask clipping's effect entirely.
			const m = 0.1;
			const v = 0.05;
			const step = 5;
			const lr = 0.1;
			const beta1 = 0.9;
			const beta2 = 0.999;
			const epsilon = 1e-7;
			const maxGradientNorm = 0.5;

			// Simulate the norm accumulated by createAccumulateGradientNormComputeNode
			// for a single parameter whose (batch-averaged) gradient is exactly
			// rawGrad: normSquared = rawGrad^2.
			const normSquared = rawGrad * rawGrad;
			// clip-by-global-norm (Pascanu et al., 2013): scale = min(1, maxNorm / norm)
			const clipScale = Math.min( 1, maxGradientNorm / Math.sqrt( normSquared ) );
			const clippedGrad = rawGrad * clipScale;

			const idx = 0;
			gpuModel.weightsBuffers.attribute.array[ idx ] = weight;
			gpuModel.weightsBuffers.attribute.needsUpdate = true;
			gpuModel.weightsBuffers.gradAttribute.array[ idx ] = Math.round( rawGrad * FIXED_POINT_SCALE );
			gpuModel.weightsBuffers.gradAttribute.needsUpdate = true;
			gpuModel.weightsBuffers.mAttribute.array[ idx ] = m;
			gpuModel.weightsBuffers.mAttribute.needsUpdate = true;
			gpuModel.weightsBuffers.vAttribute.array[ idx ] = v;
			gpuModel.weightsBuffers.vAttribute.needsUpdate = true;
			gpuModel.gradNormAttribute.array[ 0 ] = Math.round( normSquared * GRADIENT_NORM_SCALE );
			gpuModel.gradNormAttribute.needsUpdate = true;
			gpuModel.stepUniform.value = step;
			gpuModel.learningRateUniform.value = lr;
			gpuModel.invBatchUniform.value = 1;
			gpuModel.maxGradientNormUniform.value = maxGradientNorm;

			const kernel = createTextureAdamWeightsComputeNode( gpuModel, { beta1, beta2, epsilon } );
			await renderer.computeAsync( kernel );

			const gpuWeight = await readFloat( renderer, gpuModel.weightsBuffers.attribute, idx );

			const expectedClipped = referenceAdamStep( { weight, grad: clippedGrad, m, v, step, lr, beta1, beta2, epsilon } );
			const expectedUnclipped = referenceAdamStep( { weight, grad: rawGrad, m, v, step, lr, beta1, beta2, epsilon } );

			// Sanity: clipping must actually change the outcome for this input,
			// otherwise the test wouldn't be able to distinguish "clipping applied"
			// from "clipping silently skipped".
			expect( expectedClipped.weight ).not.toBeCloseTo( expectedUnclipped.weight, 3 );

			expect( gpuWeight ).toBeCloseTo( expectedClipped.weight, 4 );

		} );

	} );

	describe( 'createTextureAdamLatentsComputeNode (Adam optimizer step, latent grid buffer)', () => {

		it( 'matches the hand-derived Adam update on the latents buffer', async () => {

			const renderer = getRenderer();
			const gpuModel = createTinyGPUModel();

			const value = - 0.15;
			const grad = - 0.4;
			const m = - 0.01;
			const v = 0.002;
			const step = 12;
			const lr = 0.02;
			const beta1 = 0.9;
			const beta2 = 0.999;
			const epsilon = 1e-7;

			const idx = 2; // an arbitrary latent element within the 4-element grid
			gpuModel.latentsBuffers.attribute.array[ idx ] = value;
			gpuModel.latentsBuffers.attribute.needsUpdate = true;
			gpuModel.latentsBuffers.gradAttribute.array[ idx ] = Math.round( grad * FIXED_POINT_SCALE );
			gpuModel.latentsBuffers.gradAttribute.needsUpdate = true;
			gpuModel.latentsBuffers.mAttribute.array[ idx ] = m;
			gpuModel.latentsBuffers.mAttribute.needsUpdate = true;
			gpuModel.latentsBuffers.vAttribute.array[ idx ] = v;
			gpuModel.latentsBuffers.vAttribute.needsUpdate = true;
			gpuModel.gradNormAttribute.array[ 0 ] = 0;
			gpuModel.gradNormAttribute.needsUpdate = true;
			gpuModel.stepUniform.value = step;
			gpuModel.learningRateUniform.value = lr;
			gpuModel.invBatchUniform.value = 1;
			gpuModel.maxGradientNormUniform.value = 1000;

			const kernel = createTextureAdamLatentsComputeNode( gpuModel, { beta1, beta2, epsilon } );
			await renderer.computeAsync( kernel );

			const gpuValue = await readFloat( renderer, gpuModel.latentsBuffers.attribute, idx );
			const expected = referenceAdamStep( { weight: value, grad, m, v, step, lr, beta1, beta2, epsilon } );

			expect( gpuValue ).toBeCloseTo( expected.weight, 4 );

		} );

	} );

	describe( 'createAccumulateGradientNormComputeNode', () => {

		it( 'sums batch-averaged squared gradients across both weight and latent buffers', async () => {

			const renderer = getRenderer();
			const gpuModel = createTinyGPUModel();
			const { totalWeights, totalLatents } = gpuModel.layout;

			const invBatch = 0.5; // e.g. batchSize = 2
			const weightGrads = Array.from( { length: totalWeights }, ( _, i ) => 0.1 * ( i + 1 ) );
			const latentGrads = Array.from( { length: totalLatents }, ( _, i ) => - 0.05 * ( i + 1 ) );

			for ( let i = 0; i < totalWeights; i ++ ) {

				gpuModel.weightsBuffers.gradAttribute.array[ i ] = Math.round( weightGrads[ i ] * FIXED_POINT_SCALE );

			}

			for ( let i = 0; i < totalLatents; i ++ ) {

				gpuModel.latentsBuffers.gradAttribute.array[ i ] = Math.round( latentGrads[ i ] * FIXED_POINT_SCALE );

			}

			gpuModel.weightsBuffers.gradAttribute.needsUpdate = true;
			gpuModel.latentsBuffers.gradAttribute.needsUpdate = true;
			gpuModel.gradNormAttribute.array[ 0 ] = 0;
			gpuModel.gradNormAttribute.needsUpdate = true;
			gpuModel.invBatchUniform.value = invBatch;

			const kernel = createAccumulateGradientNormComputeNode( gpuModel );
			await renderer.computeAsync( kernel );

			const gpuNormSquared = ( await readInt( renderer, gpuModel.gradNormAttribute, 0 ) ) / GRADIENT_NORM_SCALE;

			// Independent reference: sum of squares of the batch-averaged (i.e.
			// multiplied by invBatch) gradient for every weight and every latent.
			let expectedNormSquared = 0;
			for ( const g of weightGrads ) expectedNormSquared += ( g * invBatch ) ** 2;
			for ( const g of latentGrads ) expectedNormSquared += ( g * invBatch ) ** 2;

			expect( gpuNormSquared ).toBeCloseTo( expectedNormSquared, 3 );

		} );

	} );

	describe( 'createTextureTrainBatchComputeNode (forward + L2 loss + backward micro-step)', () => {

		it( 'computes loss and weight/bias gradients matching hand-derived backprop for a single-weight linear model', async () => {

			const renderer = getRenderer();
			const gpuModel = createTinyGPUModel();
			const { layout } = gpuModel;

			// Single MLP layer (no hidden layers): weightsOffset 0 = the one
			// input->output weight, weightsOffset 1 = the bias (see
			// computeTextureModelLayout: weightsCount = inSize*outSize = 1*1 = 1,
			// followed immediately by biasesCount = 1).
			const weightLayer = layout.mlpLayers[ 0 ];
			expect( weightLayer.weightsCount ).toBe( 1 );
			expect( weightLayer.biasesCount ).toBe( 1 );

			const w = 0.6;
			const b = - 0.1;
			const latentValue = 0.3; // constant across the whole 2x2 grid, so bilinear
			// sampling at *any* UV yields exactly this value regardless of jitter -
			// this sidesteps needing to replicate the kernel's own random-UV hash
			// to predict a0.
			const target = 0.5; // constant target texture, likewise UV-independent

			gpuModel.weightsBuffers.attribute.array[ weightLayer.weightsOffset ] = w;
			gpuModel.weightsBuffers.attribute.array[ weightLayer.biasesOffset ] = b;
			gpuModel.weightsBuffers.attribute.needsUpdate = true;
			gpuModel.latentsBuffers.attribute.array.fill( latentValue );
			gpuModel.latentsBuffers.attribute.needsUpdate = true;
			gpuModel.resetLoss();

			// A single flat-colored texture: sampling anywhere returns `target`
			// in the red channel, which is all this 1-output-channel model reads.
			const texSize = 2;
			const data = new Uint16Array( texSize * texSize * 4 );
			for ( let i = 0; i < texSize * texSize; i ++ ) {

				data[ i * 4 + 0 ] = THREE.DataUtils.toHalfFloat( target );
				data[ i * 4 + 1 ] = THREE.DataUtils.toHalfFloat( target );
				data[ i * 4 + 2 ] = THREE.DataUtils.toHalfFloat( target );
				data[ i * 4 + 3 ] = THREE.DataUtils.toHalfFloat( 1 );

			}

			const sourceTexture = new THREE.DataTexture( data, texSize, texSize, THREE.RGBAFormat, THREE.HalfFloatType );
			sourceTexture.wrapS = THREE.RepeatWrapping;
			sourceTexture.wrapT = THREE.RepeatWrapping;
			sourceTexture.magFilter = THREE.LinearFilter;
			sourceTexture.minFilter = THREE.LinearFilter;
			sourceTexture.generateMipmaps = false;
			sourceTexture.needsUpdate = true;

			const kernel = createTextureTrainBatchComputeNode( gpuModel, [ sourceTexture ] );
			await renderer.computeAsync( kernel );

			// Independent first-principles reference for a single-weight linear
			// model with no output activation (channelActivations is null here,
			// so the decoder output is used directly as the prediction):
			//   z = w * a0 + b            (a0 = latentValue, constant everywhere)
			//   pred = z                  (linear, no activation)
			//   loss = 0.5 * (pred - target)^2
			//   dL/dz = (pred - target)
			//   dL/dw = dL/dz * a0
			//   dL/db = dL/dz
			const z = w * latentValue + b;
			const pred = z;
			const diff = pred - target;
			const expectedLoss = 0.5 * diff * diff;
			const expectedWeightGrad = diff * latentValue;
			const expectedBiasGrad = diff;

			const gpuLoss = await gpuModel.readLoss( renderer );
			const gpuWeightGrad = ( await readInt( renderer, gpuModel.weightsBuffers.gradAttribute, weightLayer.weightsOffset ) ) / FIXED_POINT_SCALE;
			const gpuBiasGrad = ( await readInt( renderer, gpuModel.weightsBuffers.gradAttribute, weightLayer.biasesOffset ) ) / FIXED_POINT_SCALE;

			expect( gpuLoss ).toBeCloseTo( expectedLoss, 3 );
			expect( gpuWeightGrad ).toBeCloseTo( expectedWeightGrad, 3 );
			expect( gpuBiasGrad ).toBeCloseTo( expectedBiasGrad, 3 );

			// Sign sanity, independent of the numeric tolerance above: since
			// pred (0.08) < target (0.5), the error signal must push the
			// prediction *up* - a correct gradient-descent step (weight -= lr*grad)
			// must therefore increase both the weight (positive latent input) and
			// the bias.
			expect( diff ).toBeLessThan( 0 );
			expect( gpuWeightGrad ).toBeLessThan( 0 );
			expect( gpuBiasGrad ).toBeLessThan( 0 );

			sourceTexture.dispose();

		} );

		it( 'reduces loss after one full train+adam update step (gradient descent sanity)', async () => {

			const renderer = getRenderer();
			const gpuModel = createTinyGPUModel();
			const { layout } = gpuModel;
			const weightLayer = layout.mlpLayers[ 0 ];

			gpuModel.weightsBuffers.attribute.array[ weightLayer.weightsOffset ] = 0.1;
			gpuModel.weightsBuffers.attribute.array[ weightLayer.biasesOffset ] = 0.0;
			gpuModel.weightsBuffers.attribute.needsUpdate = true;
			gpuModel.latentsBuffers.attribute.array.fill( 0.3 );
			gpuModel.latentsBuffers.attribute.needsUpdate = true;
			gpuModel.learningRateUniform.value = 0.05;
			gpuModel.stepUniform.value = 1;
			gpuModel.maxGradientNormUniform.value = 1000;

			const texSize = 2;
			const data = new Uint16Array( texSize * texSize * 4 );
			for ( let i = 0; i < texSize * texSize; i ++ ) {

				data[ i * 4 + 0 ] = THREE.DataUtils.toHalfFloat( 0.9 );
				data[ i * 4 + 1 ] = THREE.DataUtils.toHalfFloat( 0.9 );
				data[ i * 4 + 2 ] = THREE.DataUtils.toHalfFloat( 0.9 );
				data[ i * 4 + 3 ] = THREE.DataUtils.toHalfFloat( 1 );

			}

			const sourceTexture = new THREE.DataTexture( data, texSize, texSize, THREE.RGBAFormat, THREE.HalfFloatType );
			sourceTexture.wrapS = THREE.RepeatWrapping;
			sourceTexture.wrapT = THREE.RepeatWrapping;
			sourceTexture.magFilter = THREE.LinearFilter;
			sourceTexture.minFilter = THREE.LinearFilter;
			sourceTexture.generateMipmaps = false;
			sourceTexture.needsUpdate = true;

			const trainNode = createTextureTrainBatchComputeNode( gpuModel, [ sourceTexture ] );
			const adamNode = createTextureAdamWeightsComputeNode( gpuModel );

			gpuModel.resetLoss();
			await renderer.computeAsync( trainNode );
			const lossBefore = await gpuModel.readLoss( renderer );
			await renderer.computeAsync( adamNode );

			gpuModel.resetLoss();
			await renderer.computeAsync( trainNode );
			const lossAfter = await gpuModel.readLoss( renderer );

			// A single Adam step with a reasonably small learning rate must move
			// the loss downhill for a trivial, noise-free, single-sample convex
			// problem like this one (constant input, constant target, linear
			// model, L2 loss) - this is the most basic invariant of a correct
			// gradient-descent-family optimizer and would fail on e.g. a sign
			// error in the backward pass or a swapped +=/-= in the Adam update.
			expect( lossAfter ).toBeLessThan( lossBefore );

			sourceTexture.dispose();

		} );

	} );

} );

import { createNeuralTextureModel } from './NeuralTextureModel.js';
import { NeuralTextureGPUModel } from './NeuralTextureGPUModel.js';
import {
	createTextureTrainBatchComputeNode,
	createResetGradientNormComputeNode,
	createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode
} from './NeuralTextureGPUCompute.js';

const DEFAULT_OPTIONS = {
	channels: 4,
	levels: 4,
	baseResolution: 16,
	targetResolution: 256,
	hiddenSizes: [ 32, 32 ],
	outputChannels: 3,
	batchSize: 4096,
	learningRate: 0.01,
	// Anneals all the way to (near) zero by the final iteration. A learning
	// rate that never fully decays keeps Adam injecting per-step gradient
	// noise into individual grid texels indefinitely, which - since each
	// texel only sees a handful of samples per iteration - doesn't average
	// out and shows up as a noise floor that stops shrinking with more
	// training. NVIDIA's neural texture compression trainer anneals the
	// same way (cosine schedule to 0).
	cosineAnnealingScale: 0.001,
	iterations: 3000,
	maxGradientNorm: 1,
	seed: 1,
	name: 'trained neural texture'
};

/**
 * Browser-side trainer that fits a small multiresolution-grid + MLP neural
 * representation to a single static GPU texture (e.g. a material's
 * albedo/base-color channel), following the NVIDIA neural texture
 * compression / instant-ngp recipe: a trainable feature-grid positional
 * encoding feeding a shallow MLP decoder, trained with Adam + L2 loss.
 *
 * Unlike `NeuralAppearanceTrainer` (which distills an entire BRDF's shading
 * response and therefore needs a rendered "teacher atlas" + CPU readback),
 * the teacher here is already a static GPU texture, so training samples are
 * generated and read entirely on the GPU inside the training compute
 * shader - no readback round trip is needed per iteration.
 *
 * @three_import import { NeuralTextureTrainer } from 'three/addons/neural-parameters/NeuralTextureTrainer.js';
 */
class NeuralTextureTrainer {

	constructor( options = {} ) {

		this.options = { ...DEFAULT_OPTIONS, ...options };
		this.random = createRandom( this.options.seed );
		this._abortRequested = false;

	}

	/**
	 * Requests that an in-flight `train()` call finish after the current
	 * iteration and return the current model, as if the run had completed.
	 */
	abort() {

		this._abortRequested = true;

	}

	async train( { renderer, sourceTexture, onProgress = null, ...options } = {} ) {

		const settings = { ...this.options, ...options };
		this._abortRequested = false;

		if ( ! renderer || renderer.isWebGPURenderer !== true ) {

			throw new Error( 'THREE.NeuralTextureTrainer: WebGPU renderer is required for neural texture training.' );

		}

		if ( ! sourceTexture ) {

			throw new Error( 'THREE.NeuralTextureTrainer: a sourceTexture is required.' );

		}

		const cpuModel = createNeuralTextureModel( settings, this.random );
		const gpuModel = new NeuralTextureGPUModel( settings );
		gpuModel.initFromCPUModel( cpuModel );

		const trainBatchNode = createTextureTrainBatchComputeNode( gpuModel, sourceTexture );
		const resetGradientNormNode = createResetGradientNormComputeNode( gpuModel );
		const accumulateGradientNormNode = createAccumulateGradientNormComputeNode( gpuModel );
		const adamWeightsNode = createTextureAdamWeightsComputeNode( gpuModel );
		const adamLatentsNode = createTextureAdamLatentsComputeNode( gpuModel );

		const iterations = settings.iterations;
		let lastLoss = NaN;
		let completedIterations = 0;

		for ( let iteration = 0; iteration < iterations; iteration ++ ) {

			if ( this._abortRequested ) break;

			const learningRate = getLearningRate( settings, iteration );
			gpuModel.resetLoss();
			gpuModel.learningRateUniform.value = learningRate;
			gpuModel.stepUniform.value = iteration + 1;
			gpuModel.maxGradientNormUniform.value = settings.maxGradientNorm;

			renderer.compute( trainBatchNode );
			renderer.compute( resetGradientNormNode );
			renderer.compute( accumulateGradientNormNode );
			renderer.compute( adamWeightsNode );
			renderer.compute( adamLatentsNode );

			completedIterations = iteration + 1;

			const shouldSync = onProgress !== null && ( iteration % 4 === 0 || iteration === iterations - 1 );

			if ( shouldSync ) {

				lastLoss = await gpuModel.readLoss( renderer );
				await gpuModel.syncToCPU( cpuModel, renderer );

				if ( onProgress ) {

					onProgress( { iteration: completedIterations, iterations, loss: lastLoss, learningRate, cpuModel, gpuModel } );

				}

				await yieldToBrowser();

			}

			if ( iteration % 32 === 31 ) await yieldToBrowser();

		}

		await gpuModel.syncToCPU( cpuModel, renderer );

		return { cpuModel, gpuModel, loss: lastLoss, iteration: completedIterations, iterations, stoppedEarly: completedIterations < iterations };

	}

}

function getLearningRate( options, iteration ) {

	const t = Math.min( iteration / Math.max( 1, options.iterations - 1 ), 1 );
	const cosine = 0.5 * ( 1 + Math.cos( Math.PI * t ) );
	const scale = options.cosineAnnealingScale + cosine * ( 1 - options.cosineAnnealingScale );

	return options.learningRate * scale;

}

function createRandom( seed ) {

	let state = seed >>> 0;

	return function random() {

		state = ( state + 0x6D2B79F5 ) | 0;
		let value = Math.imul( state ^ state >>> 15, 1 | state );
		value ^= value + Math.imul( value ^ value >>> 7, 61 | value );

		return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

	};

}

function yieldToBrowser() {

	return new Promise( ( resolve ) => {

		if ( typeof requestAnimationFrame === 'function' ) {

			requestAnimationFrame( () => resolve() );

		} else {

			setTimeout( resolve, 0 );

		}

	} );

}

export { NeuralTextureTrainer };

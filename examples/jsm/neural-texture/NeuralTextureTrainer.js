import { createNeuralTextureModel, DEFAULT_PE_OCTAVES, DEFAULT_INPUT_ENCODING } from './NeuralTextureModel.js';
import { NeuralTextureGPUModel } from './NeuralTextureGPUModel.js';
import {
	createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode
} from './NeuralTextureGPUComputeTSL.js';
import { createResetGradientNormComputeNode } from '../neural/NeuralGPUComputeTSL.js';
import { getLearningRate, createRandom, yieldToBrowser } from '../neural/NeuralTrainingUtils.js';
import { DEFAULT_QUANTIZATION_OPTIONS, resolveQuantizationConfig, refreshGPUQuantizationRange } from '../neural/NeuralQuantization.js';

// How often (in training iterations) `quantization.range === 'auto'` is
// re-measured from the live GPU latent buffer (see
// NeuralQuantization.refreshGPUQuantizationRange). Latents move gradually
// under Adam, so the true min/max drifts slowly - a readback every 64
// iterations is frequent enough to track that drift without ever clipping
// noticeably stale, while staying rare enough (a full GPU->CPU latent-buffer
// round trip each time) not to meaningfully slow down training, which
// already only syncs that often anyway (see the `shouldSync`/`onProgress`
// cadence below).
const QUANTIZATION_RANGE_REFRESH_INTERVAL = 64;

const DEFAULT_OPTIONS = {
	channels: 4,
	levels: 4,
	baseResolution: 16,
	growthFactor: 2,
	hiddenSizes: [ 32, 32 ],
	outputChannels: 3,
	// NTC-style tiled positional-encoding octaves fed into the MLP alongside
	// the concatenated grid features - see NeuralTextureModel.js. Callers
	// that know their actual source/bake resolution up front should override
	// this with `computeTileOctaves(finestGridResolution, sourceResolution)`
	// (from NeuralGridModel.js) instead of the generic default.
	peOctaves: DEFAULT_PE_OCTAVES,
	// Which UV-derived input (if any) is fed into the MLP alongside the
	// concatenated grid features - 'none', 'raw' (u, v), or 'positional'
	// (`peOctaves` octaves of tiled positional encoding) - see
	// NeuralTextureModel.js / NeuralGridModel.getUVEncodingInputSize.
	inputEncoding: DEFAULT_INPUT_ENCODING,
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
	name: 'trained neural texture',
	// Quantization-Aware Training (QAT) of the latent grid - see
	// NeuralQuantization.js. Defaults to `mode: 'none'` (a byte-for-byte
	// no-op vs. training without QAT at all).
	quantization: DEFAULT_QUANTIZATION_OPTIONS
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
 * @three_import import { NeuralTextureTrainer } from 'three/addons/neural-texture/NeuralTextureTrainer.js';
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

	async train( { renderer, sourceTexture, sourceTextures, onProgress = null } = {} ) {

		const settings = this.options;
		this._abortRequested = false;

		if ( ! renderer || renderer.isWebGPURenderer !== true ) {

			throw new Error( 'THREE.NeuralTextureTrainer: WebGPU renderer is required for neural texture training.' );

		}

		const textures = sourceTextures || ( sourceTexture ? [ sourceTexture ] : null );

		if ( ! textures || textures.length === 0 ) {

			throw new Error( 'THREE.NeuralTextureTrainer: a sourceTexture (or sourceTextures array) is required.' );

		}

		// Validated once up front (throws a clear error immediately rather
		// than deep inside GPUModel construction) - `gpuModel` below resolves
		// it again from the same `settings.quantization` input, which is
		// idempotent (see resolveQuantizationConfig's doc comment).
		const quantization = resolveQuantizationConfig( settings );
		this.quantizationRange = null;

		const cpuModel = createNeuralTextureModel( settings, this.random );
		const gpuModel = new NeuralTextureGPUModel( settings );
		gpuModel.initFromCPUModel( cpuModel );

		try {

			const trainBatchNode = createTextureTrainBatchComputeNode( gpuModel, textures );
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

					[ lastLoss ] = await Promise.all( [
						gpuModel.readLoss( renderer ),
						gpuModel.syncToCPU( cpuModel, renderer )
					] );

					if ( onProgress ) {

						onProgress( { iteration: completedIterations, iterations, loss: lastLoss, learningRate, cpuModel, gpuModel } );

					}

					await yieldToBrowser();

				}

				// QAT `range: 'auto'` periodic re-measurement (see
				// NeuralQuantization.refreshGPUQuantizationRange and
				// QUANTIZATION_RANGE_REFRESH_INTERVAL's doc comment above).
				// Gated on `quantization` (resolved once above from
				// `settings`, not `gpuModel.quantization`) so the default
				// `mode: 'none'` path never touches `gpuModel` for this at
				// all - a true no-op, not just a cheap early-return.
				if ( quantization.mode !== 'none' && quantization.range === 'auto' &&
					( iteration % QUANTIZATION_RANGE_REFRESH_INTERVAL === QUANTIZATION_RANGE_REFRESH_INTERVAL - 1 || iteration === iterations - 1 ) ) {

					await refreshGPUQuantizationRange( gpuModel, renderer );

				}

				if ( iteration % 32 === 31 ) await yieldToBrowser();

			}

			await gpuModel.syncToCPU( cpuModel, renderer );

			// Freeze the final QAT range (see NeuralQuantization.js) so a
			// later export phase can quantize the exported latent grid against
			// exactly the range the network was actually trained/evaluated
			// against - not a range measured before the last few optimizer
			// steps moved the latents further. Retrievable as either
			// `trainer.quantizationRange` (this trainer instance) or
			// `cpuModel.quantizationRange` (the returned model) - both are the
			// same `[min, max]`-per-level array, or `null` when quantization is
			// disabled (`mode: 'none'`).
			if ( quantization.mode !== 'none' ) {

				if ( quantization.range === 'auto' ) await refreshGPUQuantizationRange( gpuModel, renderer );
				this.quantizationRange = gpuModel.getQuantizationRange();

			}

			cpuModel.quantizationRange = this.quantizationRange;

			return {
				cpuModel,
				loss: lastLoss,
				iteration: completedIterations,
				iterations,
				stoppedEarly: completedIterations < iterations,
				quantization,
				quantizationRange: this.quantizationRange
			};

		} finally {

			gpuModel.dispose();

		}

	}

}

export { NeuralTextureTrainer };

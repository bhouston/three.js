import { createGpuMaterialTeacher } from './NeuralAppearanceTeacherEvaluator.js';
import {
	createModel
} from './NeuralAppearanceModel.js';
import {
	generateTrainingSamples,
	generateIBLTrainingSamples,
	generateValidationSamples,
	normalizeDirectLightingTargets
} from './NeuralAppearanceSampler.js';
import {
	createNeuralAppearanceManifest,
	exportNeuralAppearance
} from './NeuralAppearanceManifest.js';
import { evaluateRuntimeValidation } from './NeuralAppearanceValidator.js';
import {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs
} from './NeuralAppearanceRuntime.js';
import { LEVELS, BASE_RESOLUTION, TARGET_RESOLUTION, CHANNELS_PER_LEVEL } from './NeuralAppearanceFormat.js';
import { computeGridLevels } from '../neural/NeuralGridModel.js';
import { NeuralAppearanceGPUModel } from './NeuralAppearanceGPUModel.js';
import {
	createTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createAdamWeightsComputeNode,
	createAdamLatentsComputeNode
} from './NeuralAppearanceGPUComputeTSL.js';
import {
	createResetGradientNormComputeNode,
	createResetGradientsComputeNode
} from '../neural/NeuralGPUComputeTSL.js';
import { getLearningRate, createRandom, yieldToBrowser } from '../neural/NeuralTrainingUtils.js';

const DEFAULT_OPTIONS = {
	levels: LEVELS,
	baseResolution: BASE_RESOLUTION,
	targetResolution: TARGET_RESOLUTION,
	iterations: 2000,
	iblIterations: null,
	iblTrainingRatio: 0.15,
	iblLearningRateScale: 0.35,
	batchSize: 1024,
	learningRate: 0.001,
	cosineAnnealingScale: 0.01,
	seed: 1,
	hiddenSize: 32,
	iblHiddenSize: 32,
	yieldEvery: 8,
	colorAugmentation: false,
	minimumTrainingCosine: 0.05,
	highlightLossScale: 2,
	maxGradientNorm: 1,
	previewSampleCount: 64,
	outputActivation: { type: 'linear' },
	name: 'trained neural appearance'
};

/**
 * Browser-side trainer that distills an opaque MaterialX-loaded
 * MeshPhysicalNodeMaterial into the compact neural appearance runtime format.
 *
 * @three_import import { NeuralAppearanceTrainer } from 'three/addons/neural-appearance/NeuralAppearanceTrainer.js';
 */
class NeuralAppearanceTrainer {

	constructor( options = {} ) {

		this.options = { ...DEFAULT_OPTIONS, ...options };
		this.random = createRandom( this.options.seed );
		this._abortRequested = false;

	}

	/**
	 * Requests that an in-flight `train()` call finish after the current iteration
	 * and return the current model, as if the run had completed.
	 */
	abort() {

		this._abortRequested = true;

	}

	async train( { material, renderer = null, onProgress = null, ...options } = {} ) {

		const settings = { ...this.options, ...options };
		validateTrainingSettings( settings );
		this._abortRequested = false;

		if ( ! renderer || renderer.isWebGPURenderer !== true ) {

			throw new Error( 'THREE.NeuralAppearanceTrainer: WebGPU renderer is required for neural appearance training.' );

		}

		const teacher = options.teacher || createGpuMaterialTeacher( material, renderer, {
			...settings,
			environment: options.environment || null
		} );
		let validationSamples = null;
		let directionalValidationSamples = null;
		let lastLoss = Infinity;
		let lastDirectLoss = Infinity;
		let lastIblLoss = Infinity;
		let validationLoss = Infinity;
		let validation = null;

		if ( renderer && renderer.isWebGPURenderer === true && renderer.init ) {

			await renderer.init();

		}

		if ( teacher.init ) await teacher.init();
		settings.outputFeatures = {
			emission: teacher.supportsEmission === true,
			opacity: teacher.supportsOpacity === true
		};
		settings.alphaCutoff = Number.isFinite( teacher.alphaCutoff ) ? teacher.alphaCutoff : 0.5;
		settings.opacityMode = ( options.opacityMode === 'mask' || options.opacityMode === 'blend' ) ?
			options.opacityMode :
			( ( teacher.opacityMode === 'mask' || teacher.opacityMode === 'blend' ) ? teacher.opacityMode : 'mask' );
		const model = createModel( settings, this.random );
		validationSamples = await generateTrainingSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ), colorAugmentation: false }, teacher, createRandom( settings.seed + 0x9e3779b9 ), settings.iterations );
		directionalValidationSamples = await generateValidationSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ) }, teacher );

		const gpuModel = new NeuralAppearanceGPUModel( {
			...settings,
			batchSize: settings.batchSize
		} );
		gpuModel.initFromCPUModel( model );
		const trainBatchNode = createTrainBatchComputeNode( gpuModel );
		const resetGradientNormNode = createResetGradientNormComputeNode( gpuModel );
		const resetGradientsNode = createResetGradientsComputeNode( gpuModel );
		const accumulateGradientNormNode = createAccumulateGradientNormComputeNode( gpuModel );
		const adamWeightsNode = createAdamWeightsComputeNode( gpuModel );
		const adamLatentsNode = createAdamLatentsComputeNode( gpuModel );
		const accumulateIBLGradientNormNode = createAccumulateGradientNormComputeNode( gpuModel, {
			weightOffset: gpuModel.layout.directWeightCount,
			weightCount: gpuModel.layout.iblWeightCount,
			includeLatents: false
		} );
		const adamIBLWeightsNode = createAdamWeightsComputeNode( gpuModel, {
			weightOffset: gpuModel.layout.directWeightCount,
			weightCount: gpuModel.layout.iblWeightCount
		} );
		const iblIterations = getIBLIterationCount( settings );
		const totalIterations = settings.iterations + iblIterations;

		let completedIterations = 0;

		for ( let iteration = 0; iteration < settings.iterations; iteration ++ ) {

			if ( this._abortRequested ) break;

			const lr = getLearningRate( settings, iteration );
			const samples = await generateTrainingSamples( settings, teacher, this.random, iteration );

			if ( this._abortRequested ) break;

			gpuModel.resetLoss();
			gpuModel.uploadSamples( samples, lr, iteration + 1, settings.maxGradientNorm );
			renderer.compute( trainBatchNode );
			renderer.compute( resetGradientNormNode );
			renderer.compute( accumulateGradientNormNode );
			renderer.compute( adamWeightsNode );
			renderer.compute( adamLatentsNode );

			const shouldSync = onProgress !== null || ( iteration === settings.iterations - 1 );
			if ( shouldSync ) {

				await gpuModel.syncToCPU( model, renderer );
				const losses = await gpuModel.readLosses( renderer );
				lastLoss = losses.loss;
				lastDirectLoss = losses.directLoss;
				lastIblLoss = losses.iblLoss;

				const manifest = createNeuralAppearanceManifest( model, settings );
				validation = evaluateRuntimeValidation( manifest, validationSamples, settings.previewSampleCount );
				validation.directional = evaluateRuntimeValidation( manifest, directionalValidationSamples, 0 );
				validationLoss = validation.loss;

				if ( onProgress ) {

					onProgress( {
						iteration: iteration + 1,
						iterations: totalIterations,
						phase: 'direct',
						loss: lastLoss,
						directLoss: lastDirectLoss,
						iblLoss: lastIblLoss,
						validationLoss,
						validation,
						json: manifest,
						learningRate: lr
					} );

				}

			}

			completedIterations = iteration + 1;

			if ( this._abortRequested ) break;

			if ( settings.yieldEvery > 0 && iteration % settings.yieldEvery === settings.yieldEvery - 1 ) {

				await yieldToBrowser();

			}

		}

		for ( let iblIteration = 0; iblIteration < iblIterations; iblIteration ++ ) {

			if ( this._abortRequested ) break;

			const globalIteration = settings.iterations + iblIteration;
			const lr = getLearningRate( { ...settings, iterations: iblIterations, learningRate: settings.learningRate * settings.iblLearningRateScale }, iblIteration );
			const samples = await generateIBLTrainingSamples( settings, teacher, this.random );

			if ( this._abortRequested ) break;

			gpuModel.resetLoss();
			gpuModel.uploadSamples( samples, lr, globalIteration + 1, settings.maxGradientNorm );
			renderer.compute( resetGradientsNode );
			renderer.compute( trainBatchNode );
			renderer.compute( resetGradientNormNode );
			renderer.compute( accumulateIBLGradientNormNode );
			renderer.compute( adamIBLWeightsNode );

			const shouldSync = onProgress !== null || ( iblIteration === iblIterations - 1 );
			if ( shouldSync ) {

				await gpuModel.syncToCPU( model, renderer );
				const losses = await gpuModel.readLosses( renderer );
				lastLoss = losses.loss;
				lastDirectLoss = losses.directLoss;
				lastIblLoss = losses.iblLoss;

				const manifest = createNeuralAppearanceManifest( model, settings );
				validation = evaluateRuntimeValidation( manifest, validationSamples, settings.previewSampleCount );
				validation.directional = evaluateRuntimeValidation( manifest, directionalValidationSamples, 0 );
				validationLoss = validation.loss;

				if ( onProgress ) {

					onProgress( {
						iteration: globalIteration + 1,
						iterations: totalIterations,
						phase: 'ibl',
						loss: lastLoss,
						directLoss: lastDirectLoss,
						iblLoss: lastIblLoss,
						validationLoss,
						validation,
						json: manifest,
						learningRate: lr
					} );

				}

			}

			completedIterations = globalIteration + 1;

			if ( this._abortRequested ) break;

			if ( settings.yieldEvery > 0 && iblIteration % settings.yieldEvery === settings.yieldEvery - 1 ) {

				await yieldToBrowser();

			}

		}

		if ( completedIterations > 0 ) {

			if ( validation === null ) {

				const losses = await gpuModel.readLosses( renderer );
				lastLoss = losses.loss;
				lastDirectLoss = losses.directLoss;
				lastIblLoss = losses.iblLoss;

			}

			await gpuModel.syncToCPU( model, renderer );

			const manifest = createNeuralAppearanceManifest( model, settings );
			validation = evaluateRuntimeValidation( manifest, validationSamples, settings.previewSampleCount );
			validation.directional = evaluateRuntimeValidation( manifest, directionalValidationSamples, 0 );
			validationLoss = validation.loss;

		}

		const json = await exportNeuralAppearance( model, teacher, settings );

		return {
			json,
			loss: lastLoss,
			directLoss: lastDirectLoss,
			iblLoss: lastIblLoss,
			validationLoss,
			validation,
			model,
			gpuModel,
			teacher,
			iteration: completedIterations,
			iterations: totalIterations,
			stoppedEarly: this._abortRequested
		};

	}

}

function getIBLIterationCount( settings ) {

	if ( settings.iblIterations !== null && settings.iblIterations !== undefined ) {

		return Math.max( 0, Math.floor( settings.iblIterations ) );

	}

	return Math.max( 0, Math.floor( settings.iterations * settings.iblTrainingRatio ) );

}

function validateTrainingSettings( settings ) {

	if ( settings.backend === 'cpu' ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: CPU backend training is no longer supported. Use a WebGPU renderer for training.' );

	}

	if ( Number.isInteger( settings.levels ) === false || settings.levels < 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: levels must be a positive integer.' );

	}

	if ( Number.isInteger( settings.baseResolution ) === false || settings.baseResolution < 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: baseResolution must be a positive integer.' );

	}

	if ( Number.isInteger( settings.targetResolution ) === false || settings.targetResolution < settings.baseResolution ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: targetResolution must be an integer at least baseResolution.' );

	}

	for ( const name of [ 'iterations', 'batchSize', 'hiddenSize' ] ) {

		if ( Number.isInteger( settings[ name ] ) === false || settings[ name ] < 1 ) {

			throw new Error( `THREE.NeuralAppearanceTrainer: ${ name } must be a positive integer.` );

		}

	}

	if ( settings.iblIterations !== null && settings.iblIterations !== undefined && ( Number.isInteger( settings.iblIterations ) === false || settings.iblIterations < 0 ) ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: iblIterations must be a non-negative integer.' );

	}

	if ( Number.isFinite( settings.iblTrainingRatio ) === false || settings.iblTrainingRatio < 0 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: iblTrainingRatio must be finite and non-negative.' );

	}

	if ( Number.isFinite( settings.iblLearningRateScale ) === false || settings.iblLearningRateScale <= 0 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: iblLearningRateScale must be finite and greater than zero.' );

	}

	if ( settings.outputActivation === null || settings.outputActivation === undefined || settings.outputActivation.type !== 'linear' ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Only linear output activation is supported during training.' );

	}

	if ( Number.isFinite( settings.maxGradientNorm ) === false || settings.maxGradientNorm <= 0 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: maxGradientNorm must be finite and greater than zero.' );

	}

	if ( Number.isFinite( settings.minimumTrainingCosine ) === false || settings.minimumTrainingCosine < 0 || settings.minimumTrainingCosine > 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: minimumTrainingCosine must be between zero and one.' );

	}

	if ( Number.isFinite( settings.highlightLossScale ) === false || settings.highlightLossScale < 0 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: highlightLossScale must be finite and non-negative.' );

	}

}

/**
 * Estimates GPU training-buffer bytes and exported-asset bytes for the
 * multiresolution latent grid (see NeuralGridModel.js), given the same
 * levels/baseResolution/targetResolution knobs used by `createModel`.
 */
function estimateTrainingMemory( levels = LEVELS, baseResolution = BASE_RESOLUTION, targetResolution = TARGET_RESOLUTION ) {

	const resolutions = computeGridLevels( baseResolution, targetResolution, levels );
	let latentTexels = 0;

	for ( const resolution of resolutions ) {

		latentTexels += resolution * resolution;

	}

	return {
		levels,
		baseResolution,
		targetResolution,
		resolutions,
		latentTexels,
		trainingBytes: latentTexels * CHANNELS_PER_LEVEL * 4 * 4,
		exportBytes: latentTexels * CHANNELS_PER_LEVEL * 2
	};

}

export {
	NeuralAppearanceTrainer,
	NeuralAppearanceGPUModel,
	createGpuMaterialTeacher,
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateRuntimeValidation,
	estimateTrainingMemory,
	generateTrainingSamples,
	normalizeDirectLightingTargets,
	exportNeuralAppearance
};

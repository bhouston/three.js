import { createGpuMaterialTeacher } from './NeuralAppearanceTeacherEvaluator.js';
import {
	createModel
} from './NeuralAppearanceModel.js';
import {
	generateTrainingSamples,
	generateIBLTrainingSamples,
	generateValidationSamples,
	normalizeDirectLightingTargets,
	getMipLevelCount
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
import { LATENT_CHANNELS } from './NeuralAppearanceFormat.js';
import { NeuralAppearanceGPUModel } from './NeuralAppearanceGPUModel.js';
import {
	createTrainBatchComputeNode,
	createResetGradientNormComputeNode,
	createResetGradientsComputeNode,
	createAccumulateGradientNormComputeNode,
	createAdamWeightsComputeNode,
	createAdamLatentsComputeNode
} from './NeuralAppearanceGPUCompute.js';

const DEFAULT_OPTIONS = {
	resolution: 8,
	sourceResolution: null,
	latentDownsample: 1,
	maxResolution: 4096,
	fixedTrainingMip: - 1,
	mipSamplingDecay: 0.9,
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
 * @three_import import { NeuralAppearanceTrainer } from 'three/addons/neural/NeuralAppearanceTrainer.js';
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

		const settings = resolveTrainingSettings( { ...this.options, ...options } );
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
		validationSamples = await generateTrainingSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ), colorAugmentation: false, sampleAllMips: true }, teacher, createRandom( settings.seed + 0x9e3779b9 ), settings.iterations );
		directionalValidationSamples = await generateValidationSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ) }, teacher );

		const gpuModel = new NeuralAppearanceGPUModel( {
			...settings,
			batchSize: getTrainingSampleCapacity( settings )
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

function resolveTrainingSettings( settings ) {

	const sourceResolution = settings.sourceResolution === null || settings.sourceResolution === undefined ?
		settings.resolution :
		settings.sourceResolution;
	const resolution = settings.sourceResolution === null || settings.sourceResolution === undefined ?
		settings.resolution :
		Math.max( 1, Math.floor( sourceResolution / settings.latentDownsample ) );

	return { ...settings, resolution, sourceResolution };

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

	if ( Number.isInteger( settings.resolution ) === false || settings.resolution < 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: resolution must be a positive integer.' );

	}

	if ( Number.isInteger( settings.sourceResolution ) === false || settings.sourceResolution < 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: sourceResolution must be a positive integer.' );

	}

	if ( Number.isFinite( settings.latentDownsample ) === false || settings.latentDownsample < 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: latentDownsample must be finite and at least one.' );

	}

	if ( Number.isInteger( settings.maxResolution ) === false || settings.maxResolution < 1 || settings.resolution > settings.maxResolution ) {

		throw new Error( `THREE.NeuralAppearanceTrainer: resolution must not exceed maxResolution (${ settings.maxResolution }).` );

	}

	const mipLevelCount = getMipLevelCount( settings.resolution, settings.resolution );

	if ( Number.isInteger( settings.fixedTrainingMip ) === false || settings.fixedTrainingMip < - 1 || settings.fixedTrainingMip >= mipLevelCount ) {

		throw new Error( `THREE.NeuralAppearanceTrainer: fixedTrainingMip must be -1 or a valid mip level below ${ mipLevelCount }.` );

	}

	if ( Number.isFinite( settings.mipSamplingDecay ) === false || settings.mipSamplingDecay <= 0 || settings.mipSamplingDecay > 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: mipSamplingDecay must be greater than zero and at most one.' );

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

function estimateTrainingMemory( resolution ) {

	const mipLevels = getMipLevelCount( resolution, resolution );
	let latentTexels = 0;
	let width = resolution;
	let height = resolution;

	for ( let level = 0; level < levelCount( resolution ); level ++ ) {

		latentTexels += width * height;
		width = Math.max( 1, width >> 1 );
		height = Math.max( 1, height >> 1 );

	}

	return {
		resolution,
		mipLevels,
		latentTexels,
		trainingBytes: latentTexels * LATENT_CHANNELS * 4 * 4,
		exportBytes: latentTexels * LATENT_CHANNELS * 2
	};

}

function getTrainingSampleCapacity( settings ) {

	const mipLevelCount = getMipLevelCount( settings.resolution, settings.resolution );

	return settings.fixedTrainingMip >= 0 ?
		settings.batchSize :
		settings.batchSize * mipLevelCount;

}

function levelCount( resolution ) {

	return getMipLevelCount( resolution, resolution );

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

export {
	NeuralAppearanceTrainer,
	NeuralAppearanceGPUModel,
	createGpuMaterialTeacher,
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateRuntimeValidation,
	estimateTrainingMemory,
	getTrainingSampleCapacity,
	generateTrainingSamples,
	normalizeDirectLightingTargets,
	exportNeuralAppearance
};

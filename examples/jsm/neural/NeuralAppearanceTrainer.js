import { createGpuMaterialTeacher } from './NeuralAppearanceTeacherEvaluator.js';
import {
	createModel,
	trainBatch
} from './NeuralAppearanceModel.js';
import {
	generateTrainingSamples,
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

const DEFAULT_OPTIONS = {
	resolution: 8,
	sourceResolution: null,
	latentDownsample: 1,
	maxResolution: 4096,
	fixedTrainingMip: - 1,
	mipSamplingDecay: 0.9,
	iterations: 2000,
	batchSize: 1024,
	learningRate: 0.001,
	cosineAnnealingScale: 0.01,
	seed: 1,
	hiddenSize: 32,
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

	}

	async train( { material, renderer = null, onProgress = null, ...options } = {} ) {

		const settings = resolveTrainingSettings( { ...this.options, ...options } );
		validateTrainingSettings( settings );
		const teacher = options.teacher || createGpuMaterialTeacher( material, renderer, settings );
		let validationSamples = null;
		let directionalValidationSamples = null;
		let lastLoss = Infinity;
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
		const model = createModel( settings, this.random );
		validationSamples = await generateTrainingSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ), colorAugmentation: false, sampleAllMips: true }, teacher, createRandom( settings.seed + 0x9e3779b9 ), settings.iterations );
		directionalValidationSamples = await generateValidationSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ) }, teacher );

		for ( let iteration = 0; iteration < settings.iterations; iteration ++ ) {

			const lr = getLearningRate( settings, iteration );
			const samples = await generateTrainingSamples( settings, teacher, this.random, iteration );
			lastLoss = trainBatch( model, samples, teacher, lr, iteration + 1, settings.maxGradientNorm );
			const manifest = createNeuralAppearanceManifest( model, settings );
			validation = evaluateRuntimeValidation( manifest, validationSamples, settings.previewSampleCount );
			validation.directional = evaluateRuntimeValidation( manifest, directionalValidationSamples, 0 );
			validationLoss = validation.loss;

			if ( onProgress ) {

				onProgress( {
					iteration: iteration + 1,
					iterations: settings.iterations,
					loss: lastLoss,
					validationLoss,
					validation,
					json: manifest,
					learningRate: lr
				} );

			}

			if ( settings.yieldEvery > 0 && iteration % settings.yieldEvery === settings.yieldEvery - 1 ) {

				await yieldToBrowser();

			}

		}

		const json = await exportNeuralAppearance( model, teacher, settings );

		return {
			json,
			loss: lastLoss,
			validationLoss,
			validation,
			model,
			teacher
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

function validateTrainingSettings( settings ) {

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
	createGpuMaterialTeacher,
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateRuntimeValidation,
	estimateTrainingMemory,
	generateTrainingSamples,
	normalizeDirectLightingTargets,
	exportNeuralAppearance,
	trainBatch
};

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
import { LEVELS, BASE_RESOLUTION, TARGET_RESOLUTION, CHANNELS_PER_LEVEL, resolveOpacityMode } from './NeuralAppearanceFormat.js';
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
	colorAugmentation: false,
	minimumTrainingCosine: 0.05,
	highlightLossScale: 2,
	maxGradientNorm: 1,
	previewSampleCount: 64,
	outputActivation: { type: 'linear' },
	name: 'trained neural appearance'
};

/**
 * Copies the GPU-trained weights/latents back to the CPU model, reads the
 * current loss buffer, and re-evaluates validation against a freshly
 * exported manifest - the "how are we doing right now" checkpoint both
 * training phases (and the post-training wrap-up) need identically.
 */
async function syncAndValidate( gpuModel, model, settings, validationSamples, directionalValidationSamples, renderer ) {

	const [ , losses ] = await Promise.all( [
		gpuModel.syncToCPU( model, renderer ),
		gpuModel.readLosses( renderer )
	] );

	const manifest = createNeuralAppearanceManifest( model, settings );
	const validation = evaluateRuntimeValidation( manifest, validationSamples, settings.previewSampleCount );
	validation.directional = evaluateRuntimeValidation( manifest, directionalValidationSamples, 0 );

	return {
		lastLoss: losses.loss,
		lastDirectLoss: losses.directLoss,
		lastIblLoss: losses.iblLoss,
		manifest,
		validation,
		validationLoss: validation.loss
	};

}

/**
 * Runs one training phase (the "direct" loss loop or the "IBL" loss loop) -
 * both are the same shape (generate samples, upload, run this phase's
 * compute-node sequence, periodically sync+validate+report progress, yield
 * to the browser) and previously existed as two independently hand-written
 * ~60-line copies in `train()` below that had to be kept in sync by hand.
 * `iterationOffset` is where this phase's iteration numbers start in the
 * combined direct+IBL progress count `train()` reports.
 */
async function runPhase( {
	iterationCount, generateSamples, computeLearningRate, computeNodes, phase, iterationOffset, totalIterations,
	settings, gpuModel, model, renderer, onProgress, validationSamples, directionalValidationSamples, isAborted
} ) {

	let completedIterations = iterationOffset;
	let lastResult = null;

	for ( let i = 0; i < iterationCount; i ++ ) {

		if ( isAborted() ) break;

		const lr = computeLearningRate( i );
		const samples = await generateSamples( i );

		if ( isAborted() ) break;

		gpuModel.resetLoss();
		gpuModel.uploadSamples( samples, lr, iterationOffset + i + 1, settings.maxGradientNorm );
		for ( const node of computeNodes ) renderer.compute( node );

		const shouldSync = onProgress !== null && ( i % 4 === 0 || i === iterationCount - 1 );

		if ( shouldSync ) {

			lastResult = await syncAndValidate( gpuModel, model, settings, validationSamples, directionalValidationSamples, renderer );

			if ( onProgress ) {

				onProgress( {
					iteration: iterationOffset + i + 1,
					iterations: totalIterations,
					phase,
					loss: lastResult.lastLoss,
					directLoss: lastResult.lastDirectLoss,
					iblLoss: lastResult.lastIblLoss,
					validationLoss: lastResult.validationLoss,
					validation: lastResult.validation,
					json: lastResult.manifest,
					learningRate: lr
				} );

			}

			// Mirrors NeuralTextureTrainer's loop: yield to the browser right
			// after onProgress fires, so a UI update it just triggered (e.g. a
			// loss-graph redraw) actually gets painted before the next chunk
			// of synchronous GPU work runs, instead of being queued up behind
			// it and only becoming visible once the whole phase finishes.
			await yieldToBrowser();

		}

		completedIterations = iterationOffset + i + 1;

		if ( isAborted() ) break;

		if ( i % 32 === 31 ) await yieldToBrowser();

	}

	return { completedIterations, lastResult };

}

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
		settings.opacityMode = resolveOpacityMode( options.opacityMode, teacher.opacityMode, 'mask' );
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
		const isAborted = () => this._abortRequested;
		const phaseCommon = { settings, gpuModel, model, renderer, onProgress, validationSamples, directionalValidationSamples, isAborted, totalIterations };

		let completedIterations = 0;

		const directResult = await runPhase( {
			...phaseCommon,
			iterationCount: settings.iterations,
			generateSamples: ( iteration ) => generateTrainingSamples( settings, teacher, this.random, iteration ),
			computeLearningRate: ( iteration ) => getLearningRate( settings, iteration ),
			computeNodes: [ trainBatchNode, resetGradientNormNode, accumulateGradientNormNode, adamWeightsNode, adamLatentsNode ],
			phase: 'direct',
			iterationOffset: 0
		} );

		completedIterations = directResult.completedIterations;
		if ( directResult.lastResult ) ( { lastLoss, lastDirectLoss, lastIblLoss, validation, validationLoss } = directResult.lastResult );

		const iblLearningRateSettings = { ...settings, iterations: iblIterations, learningRate: settings.learningRate * settings.iblLearningRateScale };

		const iblResult = await runPhase( {
			...phaseCommon,
			iterationCount: iblIterations,
			generateSamples: () => generateIBLTrainingSamples( settings, teacher, this.random ),
			computeLearningRate: ( iteration ) => getLearningRate( iblLearningRateSettings, iteration ),
			computeNodes: [ resetGradientsNode, trainBatchNode, resetGradientNormNode, accumulateIBLGradientNormNode, adamIBLWeightsNode ],
			phase: 'ibl',
			iterationOffset: settings.iterations
		} );

		completedIterations = iblResult.completedIterations;
		if ( iblResult.lastResult ) ( { lastLoss, lastDirectLoss, lastIblLoss, validation, validationLoss } = iblResult.lastResult );

		if ( completedIterations > 0 ) {

			( { lastLoss, lastDirectLoss, lastIblLoss, validation, validationLoss } =
				await syncAndValidate( gpuModel, model, settings, validationSamples, directionalValidationSamples, renderer ) );

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

// Declarative shape for every setting whose validity check is just "is a
// number of this kind" - covers 11 of `validateTrainingSettings`'s checks
// in one table-driven loop instead of one hand-written `if`/`throw` per
// setting. The remaining 3 (`backend`, `targetResolution` - whose minimum
// is *relative to* `baseResolution`, not a fixed bound - and
// `outputActivation` - an object shape, not a number) don't fit this shape
// and stay hand-written below.
const NUMERIC_SETTINGS_SCHEMA = [
	{ key: 'levels', kind: 'posInt' },
	{ key: 'baseResolution', kind: 'posInt' },
	{ key: 'iterations', kind: 'posInt' },
	{ key: 'batchSize', kind: 'posInt' },
	{ key: 'hiddenSize', kind: 'posInt' },
	{ key: 'iblIterations', kind: 'nullableNonNegInt' },
	{ key: 'iblTrainingRatio', kind: 'nonNegFinite' },
	{ key: 'iblLearningRateScale', kind: 'positiveFinite' },
	{ key: 'maxGradientNorm', kind: 'positiveFinite' },
	{ key: 'minimumTrainingCosine', kind: 'unitRange' },
	{ key: 'highlightLossScale', kind: 'nonNegFinite' }
];

const NUMERIC_SETTING_KINDS = {
	posInt: { check: ( v ) => Number.isInteger( v ) && v >= 1, message: 'must be a positive integer' },
	nullableNonNegInt: { check: ( v ) => v === null || v === undefined || ( Number.isInteger( v ) && v >= 0 ), message: 'must be a non-negative integer' },
	nonNegFinite: { check: ( v ) => Number.isFinite( v ) && v >= 0, message: 'must be finite and non-negative' },
	positiveFinite: { check: ( v ) => Number.isFinite( v ) && v > 0, message: 'must be finite and greater than zero' },
	unitRange: { check: ( v ) => Number.isFinite( v ) && v >= 0 && v <= 1, message: 'must be between zero and one' }
};

function validateTrainingSettings( settings ) {

	if ( settings.backend === 'cpu' ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: CPU backend training is no longer supported. Use a WebGPU renderer for training.' );

	}

	for ( const { key, kind } of NUMERIC_SETTINGS_SCHEMA ) {

		const { check, message } = NUMERIC_SETTING_KINDS[ kind ];

		if ( ! check( settings[ key ] ) ) {

			throw new Error( `THREE.NeuralAppearanceTrainer: ${ key } ${ message }.` );

		}

	}

	if ( Number.isInteger( settings.targetResolution ) === false || settings.targetResolution < settings.baseResolution ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: targetResolution must be an integer at least baseResolution.' );

	}

	if ( settings.outputActivation === null || settings.outputActivation === undefined || settings.outputActivation.type !== 'linear' ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Only linear output activation is supported during training.' );

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

/**
 * Releases the GPU-resident `teacher` and `gpuModel` a `train()` result
 * carries - every caller needs this exact two-line cleanup (the teacher's
 * atlas/readback resources and the training GPU model's storage buffers),
 * previously duplicated at each of a caller's dispose call sites.
 * `model`/`json` need no disposal - they're plain CPU data.
 */
function disposeTrainingResult( result, renderer ) {

	if ( result.teacher && result.teacher.dispose ) result.teacher.dispose();
	if ( result.gpuModel && result.gpuModel.dispose ) result.gpuModel.dispose( renderer );

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
	exportNeuralAppearance,
	disposeTrainingResult
};

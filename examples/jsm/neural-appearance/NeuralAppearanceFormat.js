const FORMAT = 'three-neural-appearance';
const VERSION = 7;

// Multiresolution latent grid encoding (shared geometry with neural-texture /
// neural-material - see NeuralGridModel.js): `LEVELS` grids geometrically
// spaced between `BASE_RESOLUTION` and `TARGET_RESOLUTION`, each contributing
// `CHANNELS_PER_LEVEL` features that are concatenated (not summed) into the
// decoder input.
const LEVELS = 4;
const BASE_RESOLUTION = 16;
const TARGET_RESOLUTION = 256;
const CHANNELS_PER_LEVEL = 4;
const LATENT_CHANNELS = LEVELS * CHANNELS_PER_LEVEL;

const DECODER_INPUT_SIZE = LATENT_CHANNELS + 12;
const IBL_INPUT_SIZE = LATENT_CHANNELS + 6;
const IBL_OUTPUT_SIZE = 4;
const INDIRECT_INPUT_SIZE = LATENT_CHANNELS + 3 + 3;
const INDIRECT_OUTPUT_SIZE = 3;
const IBL_TARGET_SIZE = 17;
const ROTATION_OUTPUT_SIZE = 12;
const DEFAULT_WRAP = 'repeat';

/**
 * `LATENT_CHANNELS`/`DECODER_INPUT_SIZE`/`IBL_INPUT_SIZE`/`INDIRECT_INPUT_SIZE`
 * above are only correct for a model actually built with `levels === LEVELS`
 * (the default). `createModel`/`NeuralAppearanceGPUModel`/
 * `createNeuralAppearanceManifest`/the runtime decoder all accept a
 * *configurable* `levels` (see e.g. webgpu_materials_neural_appearance.html's
 * "grid levels" GUI control, which offers 2/3/4/5/6/8) - for any of those,
 * the model's real latent vector is `levels * CHANNELS_PER_LEVEL` wide, not
 * the fixed `LATENT_CHANNELS` above. Every one of those consumers must derive
 * its channel/input-size math from these functions (given the model's own
 * `levels`), not from the bare constants, or it silently builds/reads a
 * decoder sized for the wrong input width - which doesn't throw, it just
 * makes every prediction `NaN` (out-of-bounds reads in a plain JS array
 * return `undefined`, and `weight * undefined` is `NaN`, which then
 * propagates through every downstream layer). That was a real, previously
 * undiscovered bug in this exact shape - see NeuralAppearanceModel.
 * levels-bug.test.js in test/vitest/ for the full writeup and a regression
 * test (now fixed).
 */
function computeLatentChannels( levels = LEVELS ) {

	return levels * CHANNELS_PER_LEVEL;

}

function computeDecoderInputSize( levels = LEVELS ) {

	return computeLatentChannels( levels ) + 12;

}

function computeIblInputSize( levels = LEVELS ) {

	return computeLatentChannels( levels ) + 6;

}

function computeIndirectInputSize( levels = LEVELS ) {

	return computeLatentChannels( levels ) + 3 + 3;

}

/**
 * Single source of truth for the model-shape options `createModel` (CPU
 * authoring) and `computeModelLayout` (GPU buffer layout) each otherwise
 * re-derive independently - `hiddenSize`, its `iblHiddenSize` clamp,
 * `baseResolution`/`targetResolution`, `levels`, and which auxiliary heads
 * are enabled. Resolving these once here means the CPU model and its GPU
 * layout can't silently disagree on a default (which previously happened:
 * `computeModelLayout` defaulted `hiddenSize` to 32 when omitted, while
 * `createModel` did not, sizing its decoder's hidden layers from
 * `options.hiddenSize` - `undefined` - directly).
 */
function resolveNeuralAppearanceModelOptions( options = {} ) {

	const hiddenSize = options.hiddenSize || 32;

	return {
		levels: options.levels || LEVELS,
		hiddenSize,
		iblHiddenSize: options.iblHiddenSize || Math.min( Math.max( hiddenSize, 16 ), 32 ),
		baseResolution: options.baseResolution || BASE_RESOLUTION,
		targetResolution: options.targetResolution || TARGET_RESOLUTION,
		supportsEmission: Boolean( options.outputFeatures && options.outputFeatures.emission ),
		supportsOpacity: Boolean( options.outputFeatures && options.outputFeatures.opacity )
	};

}

/**
 * Shared two-tier "explicit setting, or a declared value, or a default"
 * fallback for the `mask`/`blend` opacity mode - used identically by
 * `NeuralAppearanceTeacherEvaluator` (explicit option -> the material's own
 * declared mode -> a transparent/alphaTest-based heuristic) and
 * `NeuralAppearanceTrainer` (explicit option -> the teacher's already-
 * resolved mode -> `'mask'`), so the two can't independently drift on what
 * counts as a valid explicit/declared value.
 */
function resolveOpacityMode( explicit, declared, fallback ) {

	if ( explicit === 'mask' || explicit === 'blend' ) return explicit;
	if ( declared === 'mask' || declared === 'blend' ) return declared;

	return fallback;

}

export {
	FORMAT,
	VERSION,
	LEVELS,
	BASE_RESOLUTION,
	TARGET_RESOLUTION,
	CHANNELS_PER_LEVEL,
	LATENT_CHANNELS,
	DECODER_INPUT_SIZE,
	IBL_INPUT_SIZE,
	IBL_OUTPUT_SIZE,
	INDIRECT_INPUT_SIZE,
	INDIRECT_OUTPUT_SIZE,
	IBL_TARGET_SIZE,
	ROTATION_OUTPUT_SIZE,
	DEFAULT_WRAP,
	computeLatentChannels,
	computeDecoderInputSize,
	computeIblInputSize,
	computeIndirectInputSize,
	resolveNeuralAppearanceModelOptions,
	resolveOpacityMode
};

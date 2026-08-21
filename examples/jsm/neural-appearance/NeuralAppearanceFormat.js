import { getUVEncodingInputSize } from '../neural/NeuralGridModel.js';

const FORMAT = 'three-neural-appearance';
const VERSION = 8;

// Multiresolution latent grid encoding (shared geometry with neural-texture /
// neural-material - see NeuralGridModel.js): `LEVELS` grids starting at
// `BASE_RESOLUTION` and multiplying by `GROWTH_FACTOR` for each additional
// level, each contributing `CHANNELS_PER_LEVEL` features that are
// concatenated (not summed) into the decoder input.
const LEVELS = 4;
const BASE_RESOLUTION = 16;
const GROWTH_FACTOR = 2;
const CHANNELS_PER_LEVEL = 4;
const LATENT_CHANNELS = LEVELS * CHANNELS_PER_LEVEL;

// Default tiled positional-encoding octaves (see NeuralGridModel.
// triangleWaveEncode / the NTC paper's Section 4.3.2) concatenated onto the
// decoder/IBL/indirect-probe head inputs, after their existing fixed
// direction-dot values, giving those heads a way to reconstruct detail above
// the finest latent grid level's Nyquist limit - same technique as
// neural-texture/neural-material (see NeuralTextureModel.js), just applied
// to this model's 3 separate input vectors instead of 1. Defaults to 0
// (disabled, byte-for-byte the pre-existing input sizes/behavior) rather
// than neural-texture's nonzero default: unlike neural-texture, this model
// never had a raw-UV input to replace, so enabling this is a new capability
// opted into via `peOctaves`, not a fix - see NeuralAppearanceModel.js /
// NeuralAppearanceGPUComputeTSL.js for where it's threaded through.
const DEFAULT_PE_OCTAVES = 0;

// Default UV-derived input strategy for the decoder/IBL/indirect-probe heads
// - 'none' matches this model's historical (pre-`inputEncoding`) default of
// `peOctaves === 0`, i.e. no UV-derived input at all beyond the existing
// fixed direction-dot values. See `getUVEncodingInputSize` in
// NeuralGridModel.js.
const DEFAULT_INPUT_ENCODING = 'none';

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
 * and `peOctaves === DEFAULT_PE_OCTAVES` (0) - the defaults. `createModel`/
 * `NeuralAppearanceGPUModel`/`createNeuralAppearanceManifest`/the runtime
 * decoder all accept a *configurable* `levels` (see e.g.
 * webgpu_materials_neural_appearance.html's "grid levels" GUI control, which
 * offers 2/3/4/5/6/8) and `peOctaves` - for any of those, the model's real
 * input widths are `levels * CHANNELS_PER_LEVEL` (+ `peOctaves * 2` on the
 * decoder/IBL/indirect heads) wide, not the fixed constants above. Every one
 * of those consumers must derive its channel/input-size math from these
 * functions (given the model's own `levels`/`peOctaves`), not from the bare
 * constants, or it silently builds/reads a decoder sized for the wrong input
 * width - which doesn't throw, it just makes every prediction `NaN`
 * (out-of-bounds reads in a plain JS array return `undefined`, and `weight *
 * undefined` is `NaN`, which then propagates through every downstream
 * layer). That was a real, previously undiscovered bug in this exact shape
 * for `levels` - see NeuralAppearanceModel.levels-bug.test.js in
 * test/vitest/ for the full writeup and a regression test (now fixed) - the
 * same care applies to `peOctaves`.
 */
function computeLatentChannels( levels = LEVELS ) {

	return levels * CHANNELS_PER_LEVEL;

}

function computeDecoderInputSize( levels = LEVELS, peOctaves = DEFAULT_PE_OCTAVES, inputEncoding = DEFAULT_INPUT_ENCODING ) {

	return computeLatentChannels( levels ) + 12 + getUVEncodingInputSize( inputEncoding, peOctaves );

}

function computeIblInputSize( levels = LEVELS, peOctaves = DEFAULT_PE_OCTAVES, inputEncoding = DEFAULT_INPUT_ENCODING ) {

	return computeLatentChannels( levels ) + 6 + getUVEncodingInputSize( inputEncoding, peOctaves );

}

function computeIndirectInputSize( levels = LEVELS, peOctaves = DEFAULT_PE_OCTAVES, inputEncoding = DEFAULT_INPUT_ENCODING ) {

	return computeLatentChannels( levels ) + 3 + 3 + getUVEncodingInputSize( inputEncoding, peOctaves );

}

/**
 * Single source of truth for the model-shape options `createModel` (CPU
 * authoring) and `computeModelLayout` (GPU buffer layout) each otherwise
 * re-derive independently - `hiddenSize`, its `iblHiddenSize` clamp,
 * `baseResolution`/`growthFactor`, `levels`, and which auxiliary heads
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
		growthFactor: options.growthFactor || GROWTH_FACTOR,
		peOctaves: options.peOctaves !== undefined ? options.peOctaves : DEFAULT_PE_OCTAVES,
		// Which UV-derived input (if any) is fed into the decoder/IBL/indirect
		// heads alongside the concatenated latent grid features - see
		// NeuralAppearanceUVEncoding.js / NeuralGridModel.getUVEncodingInputSize.
		inputEncoding: options.inputEncoding || DEFAULT_INPUT_ENCODING,
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
	DEFAULT_PE_OCTAVES,
	DEFAULT_INPUT_ENCODING,
	BASE_RESOLUTION,
	GROWTH_FACTOR,
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

import { decodeUint8Base64, decodeFloat16Base64, decodeMLPLayersBase64 } from '../neural/NeuralBinaryCodec.js';

const FORMAT = 'three-neural-appearance';
// Bumped for the QAT + compact-binary-manifest rework (see
// NeuralAppearanceManifest.js/NeuralAppearanceLoader.js): `latents.levels[]`
// is now uint8-quantized (`dtype`/`min`/`max`/`dataBase64`, not a plain
// `data` float array) and every output head's MLP weights/biases are now
// float16-packed (`mlp: { dtype, layout, dataBase64 }`, not inline
// `layers[].weights`/`layers[].biases` arrays) - a breaking, non-backward-
// compatible change to the manifest shape, so old exports must be
// re-trained/re-exported, not just re-versioned.
const VERSION = 9;

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
 * - the default. `createModel`/`NeuralAppearanceGPUModel`/
 * `createNeuralAppearanceManifest`/the runtime decoder all accept a
 * *configurable* `levels` (see e.g. webgpu_materials_neural_appearance.html's
 * "grid levels" GUI control, which offers 2/3/4/5/6/8) - for any of those,
 * the model's real input widths are `levels * CHANNELS_PER_LEVEL` wide, not
 * the fixed constants above. Every one of those consumers must derive its
 * channel/input-size math from these functions (given the model's own
 * `levels`), not from the bare constants, or it silently builds/reads a
 * decoder sized for the wrong input width - which doesn't throw, it just
 * makes every prediction `NaN` (out-of-bounds reads in a plain JS array
 * return `undefined`, and `weight * undefined` is `NaN`, which then
 * propagates through every downstream layer). That was a real, previously
 * undiscovered bug in this exact shape for `levels` - see
 * NeuralAppearanceModel.levels-bug.test.js in test/vitest/ for the full
 * writeup and a regression test (now fixed).
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

/**
 * True iff `json` is a compact (uint8-quantized latents + float16-packed
 * MLP weights, see NeuralAppearanceManifest.js's `compactAppearanceJson`)
 * `.neuralAppearance` manifest, as opposed to the plain-float-array
 * intermediate shape `NeuralAppearanceManifest.js`'s
 * `createPlainAppearanceJson` (and every hand-rolled manifest in
 * NeuralAppearanceRuntime.test.js) uses. `decodeAppearanceManifest` below
 * is a no-op on an already-plain manifest, so callers that don't know which
 * shape they have (see NeuralAppearanceRuntime.js's evaluation entry
 * points) can check this first rather than needing two code paths.
 */
function isCompactAppearanceManifest( json ) {

	const level = json && json.latents && json.latents.levels && json.latents.levels[ 0 ];

	return Boolean( level && level.dataBase64 !== undefined );

}

/**
 * Decodes a compact `.neuralAppearance` manifest's uint8-quantized
 * `latents.levels[]` and float16-packed output head `mlp`/`rotation` blocks
 * back into the plain-float-array shape `NeuralAppearanceModel.js`'s
 * `buildDecoderInput`/`buildIBLInput`/`buildIndirectProbeInput` and
 * `NeuralAppearanceRuntime.js`'s `sampleRuntimeLatents`/
 * `evaluateDecoderLayers` already operate on - the exact inverse of
 * NeuralAppearanceManifest.js's `compactAppearanceJson`, using
 * NeuralBinaryCodec.js for every base64/float16/uint8 decode. A no-op
 * (returns `json` as-is) when `json` is already plain (see
 * `isCompactAppearanceManifest`), so callers that always want plain data can
 * call this unconditionally. Shared by `NeuralAppearanceLoader.js` (decoding
 * a loaded manifest before building runtime textures/uniforms) and
 * `NeuralAppearanceRuntime.js` (transparently accepting either shape in its
 * CPU reference evaluator, e.g. for `NeuralAppearanceValidator.js` to
 * validate a real, already-compacted export).
 */
function decodeAppearanceManifest( json ) {

	if ( isCompactAppearanceManifest( json ) === false ) return json;

	const levels = json.latents.levels.map( ( level, index ) => decodeAppearanceLevel( level, `latents.levels[${ index }]` ) );

	const outputs = {};
	for ( const key of Object.keys( json.outputs ) ) outputs[ key ] = decodeAppearanceHead( json.outputs[ key ], `outputs.${ key }` );

	return {
		...json,
		latents: { ...json.latents, levels },
		outputs
	};

}

function decodeAppearanceLevel( level, path ) {

	if ( level.dtype !== 'uint8' ) {

		throw new Error( `THREE.NeuralAppearanceFormat: Unsupported ${ path }.dtype "${ level.dtype }".` );

	}

	const expectedLength = level.width * level.height * CHANNELS_PER_LEVEL;
	const data = decodeUint8Base64( level.dataBase64, level.min, level.max, expectedLength );

	return { width: level.width, height: level.height, channels: level.channels, wrap: level.wrap, data };

}

function decodeAppearanceHead( head, path ) {

	if ( ! head.mlp || typeof head.mlp.dataBase64 !== 'string' || ! Array.isArray( head.mlp.layout ) ) {

		throw new Error( `THREE.NeuralAppearanceFormat: Manifest must define ${ path }.mlp.layout and ${ path }.mlp.dataBase64.` );

	}

	const { mlp, rotation, ...rest } = head;
	const decoded = { ...rest, layers: decodeMLPLayersBase64( mlp ) };

	if ( rotation ) {

		if ( typeof rotation.dataBase64 !== 'string' ) {

			throw new Error( `THREE.NeuralAppearanceFormat: Manifest must define ${ path }.rotation.dataBase64.` );

		}

		decoded.rotation = {
			inputSize: rotation.inputSize,
			outputSize: rotation.outputSize,
			weights: decodeFloat16Base64( rotation.dataBase64, rotation.inputSize * rotation.outputSize )
		};

	}

	return decoded;

}

export {
	FORMAT,
	VERSION,
	LEVELS,
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
	resolveOpacityMode,
	isCompactAppearanceManifest,
	decodeAppearanceManifest
};

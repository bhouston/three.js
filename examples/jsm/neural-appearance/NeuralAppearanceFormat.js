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
	DEFAULT_WRAP
};

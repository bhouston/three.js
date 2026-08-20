import { createMLP } from '../neural/NeuralMLP.js';
import { computeGridLevels, createLatentGrid, computeTileOctaves, LATENT_INIT_SCALE } from '../neural/NeuralGridModel.js';

// Default tiled positional-encoding octave count - matches NVIDIA's neural
// texture compression paper (Vaidyanathan et al. 2023, Section 4.3.2:
// `log2(8) = 3`), a reasonable default when the caller doesn't know its
// actual bake/source resolution up front. Callers that do know it should
// instead pass an explicit `peOctaves`, ideally computed via
// `computeTileOctaves(finestGridResolution, sourceResolution)` from
// NeuralGridModel.js so the encoding's frequency range actually matches the
// content being fit - see NeuralTextureTrainer.js.
const DEFAULT_PE_OCTAVES = 3;

/**
 * Single source of truth for the model-shape options shared by the CPU
 * model, the GPU training layout, and the trainer's defaults. Resolving
 * these once here - rather than each of those three re-declaring the same
 * defaults independently - means they can't silently drift apart.
 */
function resolveNeuralTextureOptions( options = {} ) {

	return {
		channels: options.channels || 4,
		levels: options.levels || 4,
		baseResolution: options.baseResolution || 16,
		growthFactor: options.growthFactor || 2,
		hiddenSizes: options.hiddenSizes || [ 32, 32 ],
		outputChannels: options.outputChannels || 3,
		// Number of NTC-style tiled triangle-wave positional-encoding octaves
		// (see NeuralGridModel.triangleWaveEncode) concatenated after the grid
		// features and fed into the MLP - replaces the raw (u, v) coordinate
		// this model used to optionally feed the decoder directly. 0 disables
		// positional encoding entirely (grid features only).
		peOctaves: options.peOctaves !== undefined ? options.peOctaves : DEFAULT_PE_OCTAVES
	};

}

/**
 * Creates the CPU-side reference model: a multiresolution feature grid pyramid
 * (instant-ngp / NVIDIA NTC style, one grid per level, features concatenated
 * across levels) feeding a small MLP decoder.
 */
function createNeuralTextureModel( options, random ) {

	const { channels, levels: requestedLevels, baseResolution, growthFactor, hiddenSizes, outputChannels, peOctaves } = resolveNeuralTextureOptions( options );

	// `computeGridLevels` may return fewer levels than requested when a
	// level's resolution would exceed `MAX_GRID_RESOLUTION` (see
	// NeuralGridModel.js) - `levels` below is reassigned to the actual grid
	// count so `inputSize` (and the returned `levels` field, which
	// `NeuralTextureGPUModel`'s layout must match) reflect what was really
	// built, not what was requested.
	const resolutions = computeGridLevels( baseResolution, growthFactor, requestedLevels );
	const levels = resolutions.length;
	const grids = resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, channels, random ) );

	// Append `peOctaves` octaves of NTC-style tiled positional encoding (2
	// values per octave: horizontal + vertical) after the concatenated grid
	// features, giving the MLP a way to reconstruct detail above the finest
	// grid level's Nyquist limit - see NeuralGridModel.triangleWaveEncode and
	// NeuralTextureGPUComputeTSL.js for how this is trained.
	const inputSize = levels * channels + peOctaves * 2;
	const decoder = createMLP( inputSize, hiddenSizes, outputChannels, random, 'relu', 'linear' );

	return { channels, levels, resolutions, grids, decoder, hiddenSizes, outputChannels, peOctaves };

}

export {
	createNeuralTextureModel,
	resolveNeuralTextureOptions,
	createLatentGrid,
	computeGridLevels,
	computeTileOctaves,
	LATENT_INIT_SCALE,
	DEFAULT_PE_OCTAVES
};

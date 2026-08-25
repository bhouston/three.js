import { createMLP } from './NTCMLP.js';
import { computeGridLevels, createLatentGrid, LATENT_INIT_SCALE } from './NTCGridModel.js';

/**
 * Single source of truth for the model-shape options shared by the CPU
 * model, the GPU training layout, and the trainer's defaults. Resolving
 * these once here - rather than each of those three re-declaring the same
 * defaults independently - means they can't silently drift apart.
 */
function resolveNTCGridPyramidOptions( options = {} ) {

	return {
		channels: options.channels || 4,
		levels: options.levels || 4,
		baseResolution: options.baseResolution || 16,
		growthFactor: options.growthFactor || 2,
		hiddenSizes: options.hiddenSizes || [ 32, 32 ],
		outputChannels: options.outputChannels || 3
	};

}

/**
 * Creates the CPU-side reference model: a multiresolution feature grid pyramid
 * (instant-ngp / NVIDIA NTC style, one grid per level, features concatenated
 * across levels) feeding a small MLP decoder.
 */
function createNTCGridPyramidModel( options, random ) {

	const { channels, levels: requestedLevels, baseResolution, growthFactor, hiddenSizes, outputChannels } = resolveNTCGridPyramidOptions( options );

	// `computeGridLevels` may return fewer levels than requested when a
	// level's resolution would exceed `MAX_GRID_RESOLUTION` (see
	// NeuralGridModel.js) - `levels` below is reassigned to the actual grid
	// count so `inputSize` (and the returned `levels` field, which
	// `NTCGPUModel`'s layout must match) reflect what was really
	// built, not what was requested.
	const resolutions = computeGridLevels( baseResolution, growthFactor, requestedLevels );
	const levels = resolutions.length;
	const grids = resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, channels, random ) );

	const inputSize = levels * channels;
	const decoder = createMLP( inputSize, hiddenSizes, outputChannels, random, 'relu', 'linear' );

	return { channels, levels, resolutions, grids, decoder, hiddenSizes, outputChannels };

}

export {
	createNTCGridPyramidModel,
	resolveNTCGridPyramidOptions,
	createLatentGrid,
	computeGridLevels,
	LATENT_INIT_SCALE
};

import { createMLP } from '../neural/NeuralMLP.js';
import { computeGridLevels, createLatentGrid, LATENT_INIT_SCALE } from '../neural/NeuralGridModel.js';

/**
 * Creates the CPU-side reference model: a multiresolution feature grid pyramid
 * (instant-ngp / NVIDIA NTC style, one grid per level, features concatenated
 * across levels) feeding a small MLP decoder.
 */
function createNeuralTextureModel( options, random ) {

	const channels = options.channels || 4;
	const levels = options.levels || 4;
	const baseResolution = options.baseResolution || 16;
	const targetResolution = options.targetResolution || 256;
	const hiddenSizes = options.hiddenSizes || [ 32, 32 ];
	const outputChannels = options.outputChannels || 3;

	const resolutions = computeGridLevels( baseResolution, targetResolution, levels );
	const grids = resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, channels, random ) );

	const inputSize = levels * channels;
	const decoder = createMLP( inputSize, hiddenSizes, outputChannels, random, 'relu', 'linear' );

	return { channels, levels, resolutions, grids, decoder, hiddenSizes, outputChannels };

}

export { createNeuralTextureModel, createLatentGrid, computeGridLevels, LATENT_INIT_SCALE };

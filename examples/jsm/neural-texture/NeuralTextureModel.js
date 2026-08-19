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
	const includeUV = options.includeUV || false;

	const resolutions = computeGridLevels( baseResolution, targetResolution, levels );
	const grids = resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, channels, random ) );

	// Optionally append the raw (u, v) sample coordinate after the concatenated
	// grid features, giving the MLP a direct "skip connection" to exact
	// sub-texel position alongside the (bilinearly-blurred) learned encoding -
	// see NeuralTextureGPUComputeTSL.js for how this is trained.
	const inputSize = levels * channels + ( includeUV ? 2 : 0 );
	const decoder = createMLP( inputSize, hiddenSizes, outputChannels, random, 'relu', 'linear' );

	return { channels, levels, resolutions, grids, decoder, hiddenSizes, outputChannels, includeUV };

}

export { createNeuralTextureModel, createLatentGrid, computeGridLevels, LATENT_INIT_SCALE };

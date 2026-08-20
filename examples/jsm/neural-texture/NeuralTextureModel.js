import { createMLP } from '../neural/NeuralMLP.js';
import { computeGridLevels, createLatentGrid, LATENT_INIT_SCALE } from '../neural/NeuralGridModel.js';

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
		targetResolution: options.targetResolution || 256,
		hiddenSizes: options.hiddenSizes || [ 32, 32 ],
		outputChannels: options.outputChannels || 3,
		// Experimental: also feed the raw (u, v) sample coordinate into the
		// MLP alongside the concatenated grid features - see below.
		includeUV: options.includeUV || false
	};

}

/**
 * Creates the CPU-side reference model: a multiresolution feature grid pyramid
 * (instant-ngp / NVIDIA NTC style, one grid per level, features concatenated
 * across levels) feeding a small MLP decoder.
 */
function createNeuralTextureModel( options, random ) {

	const { channels, levels, baseResolution, targetResolution, hiddenSizes, outputChannels, includeUV } = resolveNeuralTextureOptions( options );

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

export { createNeuralTextureModel, resolveNeuralTextureOptions, createLatentGrid, computeGridLevels, LATENT_INIT_SCALE };

import { createMLP } from './NTCMLP.js';
import { computeGridLevels, createLatentGrid, LATENT_INIT_SCALE } from './NTCGridModel.js';
import { computeAllNaturalLods } from '../NTCMipPyramid.js';

/**
 * Single source of truth for the model-shape options shared by the CPU
 * model, the GPU training layout, and the trainer's defaults. Resolving
 * these once here - rather than each of those three re-declaring the same
 * defaults independently - means they can't silently drift apart.
 *
 * `enableMipPyramid` (default `true`) turns on mip-pyramid-aware training
 * (see NTCMipPyramid.js): the finest grid levels fade out and a normalized
 * LOD value is concatenated onto the decoder's input as the requested LOD
 * coarsens, so the trained network reconstructs a correctly anti-aliased
 * result at every mip level instead of only being trained/evaluated at mip
 * 0. `textureResolution`, when mip-pyramid training is enabled, is the
 * source texture's largest dimension (falls back to the finest grid
 * resolution when omitted - the caller normally passes the real source
 * texture size, see NTCTrainer.js).
 */
function resolveNTCGridPyramidOptions( options = {} ) {

	return {
		channels: options.channels || 4,
		levels: options.levels || 4,
		baseResolution: options.baseResolution || 16,
		growthFactor: options.growthFactor || 2,
		hiddenSizes: options.hiddenSizes || [ 32, 32 ],
		outputChannels: options.outputChannels || 3,
		enableMipPyramid: options.enableMipPyramid !== undefined ? options.enableMipPyramid : true,
		textureResolution: options.textureResolution
	};

}

/**
 * Creates the CPU-side reference model: a multiresolution feature grid pyramid
 * (instant-ngp / NVIDIA NTC style, one grid per level, features concatenated
 * across levels) feeding a small MLP decoder.
 *
 * When mip-pyramid training is enabled (see `resolveNTCGridPyramidOptions`),
 * the returned model also carries a `mipPyramid` descriptor -
 * `{ textureResolution, naturalLods, maxLod }` - and the decoder's input is
 * one wider than `levels * channels` to make room for the concatenated LOD
 * value (see NTCMipPyramid.js / NTCGPUComputeTSL.js /
 * NTCDecoderTSL.js). `mipPyramid` is `null` when disabled, so a
 * consumer only needs a single truthiness check to know whether a model is
 * mip-pyramid-aware - including a model loaded from a `.ntc` file that
 * predates this feature (see NTCManifest.js / NTCLoader.js), which decodes
 * with `mipPyramid: null` and behaves exactly as before.
 */
function createNTCGridPyramidModel( options, random ) {

	const { channels, levels: requestedLevels, baseResolution, growthFactor, hiddenSizes, outputChannels, enableMipPyramid, textureResolution } = resolveNTCGridPyramidOptions( options );

	// `computeGridLevels` may return fewer levels than requested when a
	// level's resolution would exceed `MAX_GRID_RESOLUTION` (see
	// NeuralGridModel.js) - `levels` below is reassigned to the actual grid
	// count so `inputSize` (and the returned `levels` field, which
	// `NTCGPUModel`'s layout must match) reflect what was really
	// built, not what was requested.
	const resolutions = computeGridLevels( baseResolution, growthFactor, requestedLevels );
	const levels = resolutions.length;
	const grids = resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, channels, random ) );

	const mipPyramid = enableMipPyramid ? buildMipPyramidInfo( resolutions, textureResolution ) : null;

	const inputSize = levels * channels + ( mipPyramid ? 1 : 0 );
	const decoder = createMLP( inputSize, hiddenSizes, outputChannels, random, 'relu', 'linear' );

	return { channels, levels, resolutions, grids, decoder, hiddenSizes, outputChannels, mipPyramid };

}

/**
 * Builds the `mipPyramid` descriptor attached to a mip-pyramid-aware model -
 * `naturalLods` (see `computeAllNaturalLods`, one entry per grid level, in
 * the same order as `resolutions`/`grids`) and `maxLod`, the coarsest LOD
 * the model is meant to be evaluated at (the mip index of a 4x4 texel tile,
 * matching the paper's own mip-chain floor - see NTCMipPyramid.js's module
 * doc comment). `textureResolution` defaults to the finest (largest) grid
 * resolution when not supplied - a reasonable stand-in when the caller
 * doesn't know the real source texture size, though `NTCTrainer.js` always
 * passes the true one.
 */
function buildMipPyramidInfo( resolutions, textureResolution ) {

	const resolvedTextureResolution = textureResolution || resolutions[ resolutions.length - 1 ];
	const naturalLods = computeAllNaturalLods( resolutions, resolvedTextureResolution );
	const maxLod = Math.max( 1, Math.ceil( Math.log2( Math.max( 1, resolvedTextureResolution ) ) ) );

	return { textureResolution: resolvedTextureResolution, naturalLods, maxLod };

}

export {
	createNTCGridPyramidModel,
	resolveNTCGridPyramidOptions,
	createLatentGrid,
	computeGridLevels,
	LATENT_INIT_SCALE
};

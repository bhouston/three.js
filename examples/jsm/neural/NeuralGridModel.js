const LATENT_INIT_SCALE = 0.35;

// Single source of truth for the acceptable/offered values of the four grid
// + MLP shape options (grid levels, grid base resolution, per-level growth
// factor, MLP hidden width) - shared by every example's GUI dropdown
// (neural-texture, neural-material, neural-appearance) so their offered
// ranges can't silently drift apart. These are display/GUI option lists, not
// hard validation: `computeGridLevels` itself accepts any positive
// `baseResolution`/`growthFactor`/`levels` combination, not just these.
const GRID_LEVELS_OPTIONS = [ 1, 2, 3, 4, 5, 6, 8 ];
const GRID_BASE_RESOLUTION_OPTIONS = [ 4, 8, 16, 32, 64 ];
const GRID_GROWTH_FACTOR_OPTIONS = [ 2, 3, 4, 5 ];
const MLP_HIDDEN_SIZE_OPTIONS = [ 8, 16, 32, 64, 128 ];

// Hard ceiling on any single grid level's resolution - independent of the GUI
// option lists above, this is enforced by `computeGridLevels` itself so no
// caller (GUI-driven or programmatic) can ever request a latent grid texture
// large enough to be impractical (a 4096x4096 level alone is already 64x the
// texel count of a 512x512 one).
const MAX_GRID_RESOLUTION = 4096;

/**
 * Computes the per-level grid resolutions for a multiresolution feature grid
 * (instant-ngp / NVIDIA neural texture compression style). Starts at
 * `baseResolution` and multiplies by `growthFactor` for each additional
 * level, e.g. `computeGridLevels( 16, 2, 4 )` -> `[ 16, 32, 64, 128 ]`. When
 * `levels <= 1` this just returns `[ baseResolution ]` - there's no separate
 * "target resolution" to special-case. Resolutions are always clamped to
 * >= 1 and rounded to integers. Any level that would exceed
 * `MAX_GRID_RESOLUTION` is simply omitted rather than clamped down to it -
 * since resolution only grows (or stays flat) from one level to the next,
 * once a level is dropped every subsequent one would exceed the cap too, so
 * the returned array may be shorter than `levels` (or, in the degenerate
 * case of a `baseResolution` already over the cap, empty). Shared by every
 * trainer that uses a multiresolution grid positional encoding
 * (neural-texture, neural-material, neural-appearance) so the geometry never
 * drifts between them.
 */
function computeGridLevels( baseResolution, growthFactor, levels ) {

	const resolutions = [];
	const count = Math.max( 1, Math.round( levels ) );
	let resolution = Math.max( 1, Math.round( baseResolution ) );

	for ( let i = 0; i < count; i ++ ) {

		if ( resolution > MAX_GRID_RESOLUTION ) break;

		resolutions.push( resolution );
		resolution = Math.max( 1, Math.round( resolution * growthFactor ) );

	}

	return resolutions;

}

function createLatentGrid( width, height, channels, random ) {

	const data = new Float32Array( width * height * channels );

	for ( let i = 0; i < data.length; i ++ ) {

		data[ i ] = ( random() * 2 - 1 ) * LATENT_INIT_SCALE;

	}

	return { width, height, channels, data };

}

/**
 * How many octaves of tiled positional encoding (see `triangleWaveEncode`
 * below) are needed to bridge the frequency gap between a grid pyramid's
 * finest level and the resolution content is actually being reconstructed
 * at. Mirrors NVIDIA's neural texture compression paper (Vaidyanathan et
 * al. 2023, Section 4.3.2): their finest grid can be up to 8x coarser than
 * the texel resolution it reconstructs, so they use `log2(8) = 3` octaves -
 * exactly enough to span that one grid cell's upsampling factor, since every
 * lower frequency is already carried by the learned grid features
 * themselves. `targetTexelResolution` should be the actual resolution
 * content is being fit/rendered at (e.g. the source texture's resolution,
 * or a material's bake resolution) - not the grid's own finest level
 * resolution.
 */
function computeTileOctaves( finestGridResolution, targetTexelResolution ) {

	const ratio = Math.max( 1, targetTexelResolution ) / Math.max( 1, finestGridResolution );

	return Math.max( 0, Math.ceil( Math.log2( Math.max( 1, ratio ) ) ) );

}

/**
 * Triangle wave in [-1, 1] with period 1, matching the computationally
 * cheaper triangle-wave positional encoding NTC uses in place of sin/cos
 * (Section 4.3.2: "we observe no quality loss" versus the trigonometric
 * variant). Kept in exact lockstep with `triangleWaveTSL` below - any change
 * here must be mirrored there, and vice versa, or CPU authoring/reference
 * evaluation and GPU training/inference will silently disagree.
 */
function triangleWave( x ) {

	const wrapped = x - Math.floor( x + 0.5 );

	return 4 * Math.abs( wrapped ) - 1;

}

/**
 * NTC-style tiled positional encoding (Vaidyanathan et al. 2023, Section
 * 4.3.2 / Fig. 5): `octaves` increasing-frequency triangle waves per axis,
 * concatenated as `octaves` horizontal values followed by `octaves`
 * vertical values (2*octaves scalars total - mirrors the paper's "6+6
 * scalars" example for 3 octaves). Deliberately *local/tiled* - each octave
 * `o` repeats every `1 / 2^o` of the UV unit square, so the whole encoding
 * repeats every `2^(octaves-1)` UV-space texels of its *finest* octave -
 * rather than spanning the full [0,1) UV range like a global Fourier
 * encoding would. That's intentional: its only job is to help the decoder
 * reconstruct detail *above* the finest grid level's Nyquist limit: every
 * lower frequency is already carried by the learned grid features it's
 * concatenated alongside (see e.g. NeuralAppearanceModel.sampleLatents /
 * the grid taps in NeuralTextureGPUComputeTSL.js), so this encoding must
 * not (and does not) replace the grid - only supplement it.
 */
function triangleWaveEncode( u, v, octaves ) {

	const values = new Array( octaves * 2 );

	for ( let o = 0; o < octaves; o ++ ) {

		const frequency = Math.pow( 2, o );
		values[ o ] = triangleWave( u * frequency );
		values[ octaves + o ] = triangleWave( v * frequency );

	}

	return values;

}

export {
	computeGridLevels,
	createLatentGrid,
	computeTileOctaves,
	triangleWave,
	triangleWaveEncode,
	LATENT_INIT_SCALE,
	GRID_LEVELS_OPTIONS,
	GRID_BASE_RESOLUTION_OPTIONS,
	GRID_GROWTH_FACTOR_OPTIONS,
	MLP_HIDDEN_SIZE_OPTIONS,
	MAX_GRID_RESOLUTION
};

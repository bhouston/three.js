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

export {
	computeGridLevels,
	createLatentGrid,
	LATENT_INIT_SCALE,
	GRID_LEVELS_OPTIONS,
	GRID_BASE_RESOLUTION_OPTIONS,
	GRID_GROWTH_FACTOR_OPTIONS,
	MLP_HIDDEN_SIZE_OPTIONS,
	MAX_GRID_RESOLUTION
};

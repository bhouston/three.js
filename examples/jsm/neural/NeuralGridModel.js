const LATENT_INIT_SCALE = 0.35;

/**
 * Computes the per-level grid resolutions for a multiresolution feature grid,
 * geometrically spaced between a coarse base resolution and a fine target
 * resolution (instant-ngp / NVIDIA neural texture compression style). Shared
 * by every trainer that uses a multiresolution grid positional encoding
 * (neural-texture, neural-material, neural-appearance) so the geometry never
 * drifts between them.
 */
function computeGridLevels( baseResolution, targetResolution, levels ) {

	const resolutions = [];

	if ( levels <= 1 ) {

		resolutions.push( Math.max( 1, Math.round( targetResolution ) ) );

	} else {

		const growth = Math.pow( Math.max( targetResolution, baseResolution ) / Math.max( 1, baseResolution ), 1 / ( levels - 1 ) );

		for ( let i = 0; i < levels; i ++ ) {

			resolutions.push( Math.max( 1, Math.round( baseResolution * Math.pow( growth, i ) ) ) );

		}

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

export { computeGridLevels, createLatentGrid, LATENT_INIT_SCALE };

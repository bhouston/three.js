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
 * or a material's bake resolution) - not the grid's own `targetResolution`.
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
	LATENT_INIT_SCALE
};

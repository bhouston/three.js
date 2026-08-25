import { clamp, float } from 'three/tsl';

// Shared mip-pyramid math for NTC's multiresolution feature grid - used by
// both the training compute kernel (NeuralTextureGPUComputeTSL.js /
// NTCGridPyramidModel.js) and the runtime decoder (NTCDecoderTSL.js), so the
// "which grid levels matter at this LOD" rule can never drift between the
// two. Kept in this parent (non-`training/`) folder, like NTCFormat.js, since
// the runtime decode path needs it without pulling in any training-only
// dependency.
//
// Context: `NTCGridModel.computeGridLevels` builds a plain geometric
// progression of grid resolutions used as a multiresolution positional
// encoding (instant-ngp style) - by itself that encoding has no notion of
// "this texture is being viewed from far away, so its highest-frequency
// content can never resolve". Trained and sampled identically regardless of
// viewing distance, it aliases once the rendered footprint of a texel
// shrinks below a pixel - exactly the failure mode described in the NVIDIA
// neural texture compression paper (Vaidyanathan et al. 2023) that Table 1's
// per-mip-band feature levels solve. The functions below add that missing
// LOD-awareness on top of the existing geometric-progression grid model:
// every level keeps its resolution/position in `computeGridLevels`'s
// progression unchanged, but the highest-resolution (highest-frequency)
// levels fade their contribution to zero once the requested LOD is coarser
// than what they can usefully represent - both during training (so the
// network learns to reconstruct correctly *without* relying on those levels
// at coarse LOD) and at inference (so a level whose contribution is zero can
// also be skipped - see NTCDecoderTSL.js).

/**
 * Width of the linear fade-out band, in mip levels, once a grid level's LOD
 * contribution starts dropping (see `computeLevelLodWeight`). A one-mip-wide
 * band - rather than a hard cutoff - avoids a visible "pop" as the camera
 * moves and a feature level's contribution crosses its natural LOD, matching
 * the paper's own caution (Section 5.3) about abrupt LOD transitions.
 */
const FADE_BAND_MIPS = 1;

/**
 * The "natural" LOD (mip index) at which a grid level of `gridResolution`
 * stops adding resolvable detail for a texture of `textureResolution` (its
 * largest dimension): the mip index at which the target's own resolution
 * has fallen to (or below) this level's resolution. Below this LOD (closer
 * to the camera / finer mips) the target texture still has more detail than
 * this grid level encodes, so the level is useful; above it, the target mip
 * is already blurrier than what this level represents, so the level's
 * contribution should fade toward zero (see `computeLevelLodWeight`) rather
 * than injecting high-frequency content the current mip can't resolve
 * (aliasing).
 */
function computeNaturalLod( gridResolution, textureResolution ) {

	return Math.max( 0, Math.log2( Math.max( 1, textureResolution ) / Math.max( 1, gridResolution ) ) );

}

/** `computeNaturalLod` applied to every entry of `gridResolutions`. */
function computeAllNaturalLods( gridResolutions, textureResolution ) {

	return gridResolutions.map( ( resolution ) => computeNaturalLod( resolution, textureResolution ) );

}

/**
 * Plain-JS (CPU-side) continuous fade weight in [0, 1] for a grid level whose
 * natural LOD is `naturalLod`, evaluated at `lod`: full weight (1) at or
 * below `naturalLod`, fading linearly to 0 over `FADE_BAND_MIPS` once `lod`
 * exceeds it. Shared formula for both the trainer's CPU-side bookkeeping and
 * tests - see `computeLevelLodWeightTSL` for the GPU/node-graph equivalent
 * used inside an actual training/inference kernel, which must compute the
 * identical value from node inputs rather than plain numbers.
 */
function computeLevelLodWeight( lod, naturalLod ) {

	const t = ( lod - naturalLod ) / FADE_BAND_MIPS;

	return Math.min( 1, Math.max( 0, 1 - t ) );

}

/**
 * TSL node-graph equivalent of `computeLevelLodWeight` - `naturalLod` is a
 * plain JS number (baked into the generated shader as a literal, since it's
 * fixed once the grid's resolutions/textureResolution are known at model-
 * build time), `lodNode` is the runtime LOD (a TSL float node, e.g. derived
 * from screen-space UV derivatives or an explicit per-sample training LOD).
 */
function computeLevelLodWeightTSL( lodNode, naturalLod ) {

	return clamp( float( 1 ).sub( lodNode.sub( naturalLod ).div( FADE_BAND_MIPS ) ), 0, 1 );

}

export {
	FADE_BAND_MIPS,
	computeNaturalLod,
	computeAllNaturalLods,
	computeLevelLodWeight,
	computeLevelLodWeightTSL
};

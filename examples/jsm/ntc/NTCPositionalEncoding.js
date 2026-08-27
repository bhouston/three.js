import { fract } from 'three/tsl';

// Optional decoder-input positional encoding (see the NVIDIA neural texture
// compression paper, Section 4.3.2 / Fig. 5) - an opt-in extra input to the
// MLP decoder that encodes a texel's own sub-cell (fractional) position
// within its selected grid level, using triangle waves (a cheaper
// alternative to sin/cos the paper itself adopts, citing Müller et al. 2021
// - see NTCGridPyramidModel.js's `positionalEncoding` option).
//
// This addon samples its feature grids with ordinary hardware/software
// bilinear interpolation (see NTCDecoderTSL.js/NTCGPUComputeTSL.js), not the
// paper's "learned interpolation" (four raw, unfiltered corner taps
// concatenated into the MLP) - so positional encoding here serves a
// narrower purpose than in the paper: since a bilinearly-*filtered* feature
// vector alone carries no information about exactly where within its texel
// cell a query point falls, giving the decoder that sub-texel position
// directly lets it learn a position-dependent residual/sharpening
// correction it otherwise has no way to produce (a filtered feature, by
// construction, maps to the same decoded value everywhere within a cell).
// This is a real, testable capability independent of learned interpolation
// - see this repo's plan docs (.cursor/plans/ntc_paper_gap_02_*) for why
// it's introduced on its own, ahead of (and without depending on) any
// change to how corners are sampled.
//
// Each octave contributes 4 scalars - a triangle wave and its
// quarter-period-shifted twin (the triangle-wave analogue of a sin/cos
// pair), for each of the two UV axes - so `positionalEncodingSize(octaves)`
// is `4 * octaves`.

/**
 * A periodic triangle wave: period 1, range [-1, 1], `x` any real number
 * (only its fractional part matters). Cheaper than `sin`/`cos` (no
 * transcendental function, just an abs/multiply/add), while still supplying
 * the same kind of smoothly-varying phase information a coordinate network
 * needs - see the module doc comment above and Müller et al. 2021 (cited by
 * the NVIDIA neural texture compression paper, Section 4.3.2) for the
 * general technique.
 */
function triangleWave( x ) {

	const t = x - Math.floor( x );

	return 1 - 4 * Math.abs( t - 0.5 );

}

/**
 * TSL mirror of `triangleWave` - must match it exactly, the same way every
 * other CPU/TSL pair in this addon does (see NTCMLP.js's hardGELU /
 * NTCMLPTSL.js's hardGeluTSL for the analogous relationship).
 */
function triangleWaveTSL( xNode ) {

	const t = fract( xNode );

	return t.sub( 0.5 ).abs().mul( - 4 ).add( 1 );

}

/**
 * Total scalar count contributed by positional encoding at a given octave
 * count - see the module doc comment (4 values per octave: 2 axes x 2
 * phases). Shared by every place that needs to size a decoder's input
 * vector (NTCGridPyramidModel.js, NTCGPUModel.js) or a manifest field
 * (NTCManifest.js/NTCLoader.js) from `octaves` alone, so they can't
 * independently drift on the "4x" factor.
 */
function positionalEncodingSize( octaves ) {

	return 4 * octaves;

}

/**
 * Plain-JS reference implementation - `(tx, ty)` is the texel's fractional
 * position (each in `[0, 1)`) within its selected grid level's cell (the
 * same `tx`/`ty` an ordinary bilinear sample already computes as its
 * interpolation weights - see NTCGPUComputeTSL.js), `octaves` the number of
 * frequency doublings to encode. Returns a flat array of
 * `positionalEncodingSize(octaves)` values, ordered
 * `[octave0: tx-phase0, tx-phase1, ty-phase0, ty-phase1, octave1: ...]` -
 * `computePositionalEncodingTSL` below must build its output in this exact
 * order, since both are concatenated onto the decoder's input vector at a
 * fixed offset (see NTCGridPyramidModel.js's `inputSize`).
 */
function computePositionalEncoding( tx, ty, octaves ) {

	const values = new Array( positionalEncodingSize( octaves ) );

	for ( let h = 0; h < octaves; h ++ ) {

		const frequency = 2 ** h;
		const base = h * 4;

		values[ base ] = triangleWave( tx * frequency );
		values[ base + 1 ] = triangleWave( tx * frequency + 0.25 );
		values[ base + 2 ] = triangleWave( ty * frequency );
		values[ base + 3 ] = triangleWave( ty * frequency + 0.25 );

	}

	return values;

}

/**
 * TSL mirror of `computePositionalEncoding` - `txNode`/`tyNode` are TSL
 * scalar float nodes, `octaves` a plain JS integer (unrolled at kernel-build
 * time, exactly like every other "how many levels/layers" count in this
 * addon - see e.g. NTCGPUComputeTSL.js's `for (let g = 0; g < gridLevels.
 * length; g++)`). Returns a flat JS array of `positionalEncodingSize(octaves)`
 * TSL scalar nodes, in the same order as the CPU version above.
 */
function computePositionalEncodingTSL( txNode, tyNode, octaves ) {

	const values = new Array( positionalEncodingSize( octaves ) );

	for ( let h = 0; h < octaves; h ++ ) {

		const frequency = 2 ** h;
		const base = h * 4;

		values[ base ] = triangleWaveTSL( txNode.mul( frequency ) );
		values[ base + 1 ] = triangleWaveTSL( txNode.mul( frequency ).add( 0.25 ) );
		values[ base + 2 ] = triangleWaveTSL( tyNode.mul( frequency ) );
		values[ base + 3 ] = triangleWaveTSL( tyNode.mul( frequency ).add( 0.25 ) );

	}

	return values;

}

export {
	triangleWave,
	triangleWaveTSL,
	positionalEncodingSize,
	computePositionalEncoding,
	computePositionalEncodingTSL
};

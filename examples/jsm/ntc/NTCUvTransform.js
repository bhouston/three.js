import { cos, exp, float, sin, vec2 } from 'three/tsl';

// A learned, per-material 2D affine UV transform, applied to the *grid-
// indexing* coordinate only (never to a ground-truth sample - see
// NTCGPUComputeTSL.js/NTCNodeMaterial.js), so a fixed-resolution axis-aligned
// feature grid can be re-aligned to content that's rotated/scaled relative to
// the mesh's raw UV. Kept in this parent (non-`training/`) folder, like
// NTCMipBands.js/NTCFormat.js, since both the training kernel and the
// runtime decoder need the same TSL compose/apply logic and must never
// disagree about it.
//
// Runtime only ever consumes an already-baked 6-float matrix (see
// `composeUvTransformMatrix` below) - stored flat as `[a, b, c, d, e, f]`
// meaning:
//   u' = a*u + b*v + e
//   v' = c*u + d*v + f
// i.e. `[u' v'] = [u v 1] . M` for a 3x2 matrix `M = [[a,b],[c,d],[e,f]]`.
// Training keeps a decomposed (rotation, scale) form instead (see
// NTCGridPyramidModel.js) - decomposed parameters are far more stable to
// optimize than raw 6-DOF regression (identity-init, no risk of a near-
// singular/flipped matrix mid-training - standard Spatial Transformer
// Networks practice, Jaderberg et al. 2015) - and only composes to this flat
// matrix form once, at export time.

const IDENTITY_UV_TRANSFORM = [ 1, 0, 0, 1, 0, 0 ];

/**
 * Composes a decomposed (rotation, scale[, translation]) UV transform into
 * the flat 6-float matrix form above - plain JS, used once at export time
 * (see NTCManifest.js) to bake the final trained transform, and by tests.
 *
 * `rotation` is radians. `scale` is `[sx, sy]` (already-exponentiated,
 * i.e. the actual multiplicative scale - callers optimizing in log-space,
 * see NTCGridPyramidModel.js's doc comment on why, exponentiate before
 * calling this). `translation` defaults to `[0, 0]` (v1 trains rotation +
 * anisotropic scale only - see NTCGridPyramidModel.js's doc comment on why
 * translation is left out: it's close to a free/degenerate DOF under this
 * addon's default `repeat` grid wrap for a single global transform).
 *
 * Composition order is scale-then-rotate-then-translate (`M = T . R . S`),
 * the conventional 2D affine decomposition order.
 */
function composeUvTransformMatrix( rotation, scale, translation = [ 0, 0 ] ) {

	const [ sx, sy ] = scale;
	const [ tx, ty ] = translation;
	const c = Math.cos( rotation );
	const s = Math.sin( rotation );

	// `+ 0` normalizes away a `-0` result (e.g. rotation 0 => `-Math.sin(0) *
	// sy === -0`) - harmless numerically, but `-0` fails a naive `toEqual`
	// against a literal `0` in tests, and there's no reason to expose it.
	return [ c * sx, ( - s * sy ) + 0, s * sx, c * sy, tx, ty ];

}

/**
 * TSL node-graph equivalent of `composeUvTransformMatrix`, for training's
 * live per-step matrix (rebuilt every invocation from the current, still-
 * being-optimized `rotationNode`/`logScaleNode` uniforms - see
 * NTCTrainer.js's SPSA outer loop) - returns the same flat 6-entry
 * `[a,b,c,d,e,f]` shape, but as TSL float nodes instead of plain numbers.
 *
 * `logScaleNode` is a `vec2` node holding `log(scale)`, not `scale` itself -
 * exponentiated internally here - so the outer loop can optimize scale in
 * log-space (guarantees positivity, symmetric gradient behavior around 1x -
 * standard practice, see NTCGridPyramidModel.js's doc comment) without every
 * caller having to remember to exponentiate first. Translation is always `0`
 * here (see this module's doc comment above).
 */
function composeUvTransformMatrixTSL( rotationNode, logScaleNode ) {

	const c = cos( rotationNode );
	const s = sin( rotationNode );
	const sx = exp( logScaleNode.x );
	const sy = exp( logScaleNode.y );

	return [ c.mul( sx ), s.mul( sy ).negate(), s.mul( sx ), c.mul( sy ), float( 0 ), float( 0 ) ];

}

/**
 * Applies a flat 6-entry affine matrix (see this module's doc comment) to a
 * TSL `vec2` coordinate node - `matrix` entries may be plain JS numbers
 * (the common runtime case: a baked matrix loaded straight from a `.ntc`
 * manifest, see NTCNodeMaterial.js) or TSL nodes (training's live,
 * per-step-rebuilt matrix from `composeUvTransformMatrixTSL` above) - mixing
 * both in the same call is fine.
 */
function applyUvTransformTSL( coord, matrix ) {

	const [ a, b, c, d, e, f ] = matrix.map( ( v ) => ( typeof v === 'number' ? float( v ) : v ) );

	const u = coord.x.mul( a ).add( coord.y.mul( b ) ).add( e );
	const v = coord.x.mul( c ).add( coord.y.mul( d ) ).add( f );

	return vec2( u, v );

}

/**
 * Normalizes any `cpuModel.uvTransform` value to the flat 6-float matrix
 * form (or `null`) - `cpuModel.uvTransform` can legitimately be in one of
 * three shapes depending on when it's read:
 *
 *  - `null`/`undefined` (no transform: `enableUvTransform` was never set, or
 *    a `.ntc` file predating this feature) - passes through as `null`.
 *  - the *decomposed* `{ rotation, scale }` form `createNTCGridPyramidModel`
 *    initializes it to (see NTCGridPyramidModel.js) - still true for any
 *    `cpuModel` a training `onProgress` callback sees mid-training (see
 *    NTCTrainer.js: the decomposed form is only baked to the flat matrix
 *    once, right after the loop ends), so a live preview built from
 *    `progress.cpuModel` needs this composed on demand.
 *  - the already-*baked* flat 6-float array (post-training, or loaded from a
 *    `.ntc` manifest, see NTCManifest.js/NTCLoader.js) - passed through
 *    as-is.
 *
 * Every consumer of `cpuModel.uvTransform` as an actual matrix (
 * NTCNodeMaterial.js, NTCManifest.js's `encodeNTC`) should go through this
 * rather than assume the flat-array shape directly, so a live mid-training
 * preview and a final export both work the same way.
 */
function resolveUvTransformMatrix( uvTransform ) {

	if ( ! uvTransform ) return null;
	if ( Array.isArray( uvTransform ) ) return uvTransform;

	return composeUvTransformMatrix( uvTransform.rotation, uvTransform.scale );

}

export {
	IDENTITY_UV_TRANSFORM,
	composeUvTransformMatrix,
	composeUvTransformMatrixTSL,
	applyUvTransformTSL,
	resolveUvTransformMatrix
};

/**
 * Detects whether the current environment's WGSL compiler supports the
 * "Packed 4x8 Integer Dot Product" WGSL language feature (`dot4I8Packed`/
 * `dot4U8Packed`/`pack4xI8`/`pack4xU8`/`unpack4xI8`/`unpack4xU8`) - shared by
 * {@link PackedDotProductNode}, {@link PackIntegerNode} and
 * {@link UnpackIntegerNode}, all of which need this same check before
 * emitting a call to one of those builtins on the WebGPU backend.
 *
 * This is a WGSL *language feature*, not an *extension*: it needs no
 * `enable` directive, so - unlike a device feature (e.g. `shader-f16`,
 * checked via `renderer.hasFeature(...)`/`device.features`) - it's checked
 * via `navigator.gpu.wgslLanguageFeatures.has(...)`, a property of the `GPU`
 * object itself, available before ever requesting an adapter/device. See
 * `examples/jsm/ntc/NTCPackedDotProduct.js`'s module doc comment for the
 * full background on this distinction, confirmed empirically during that
 * addon's own development.
 *
 * Support is real, but not universal: it's part of WebGPU's *optional*
 * language feature set (unlike, say, basic arithmetic), so a given browser's
 * WGSL compiler is not guaranteed to implement it just because it implements
 * WebGPU in general. Without this check, a WebGPU backend lacking the
 * feature would fail to compile *any* shader using these builtins outright
 * (an "unresolved call target" error, confirmed directly against Chromium
 * when probing a since-abandoned `pack4xI8Clamped`, which genuinely isn't a
 * WGSL builtin) - there is no automatic native-side fallback the way the
 * WebGL backend's GLSL polyfill provides one. Each of the three nodes above
 * uses this check to fall back to a hand-written TSL polyfill (built from
 * ordinary bitwise operators, themselves core WGSL/GLSL functionality) when
 * running on WebGPU without the feature, rather than assuming WebGPU always
 * implies support.
 *
 * The result is cached at module scope - `navigator.gpu.wgslLanguageFeatures`
 * is a static, environment-wide capability that can't change over the
 * lifetime of a page, so there's no reason to re-check it on every node
 * build.
 *
 * @returns {boolean} Whether `packed_4x8_integer_dot_product` is supported.
 */

let cachedResult = null;
let testOverride;

function supportsPackedIntegerDotProductFeature() {

	if ( testOverride !== undefined ) return testOverride;

	if ( cachedResult === null ) {

		cachedResult = Boolean(
			typeof navigator !== 'undefined' &&
			navigator.gpu &&
			navigator.gpu.wgslLanguageFeatures &&
			navigator.gpu.wgslLanguageFeatures.has( 'packed_4x8_integer_dot_product' )
		);

	}

	return cachedResult;

}

/**
 * Test-only hook: forces `supportsPackedIntegerDotProductFeature()` to
 * return `value` regardless of the real environment, so the WebGPU-without-
 * the-feature fallback path in `PackedDotProductNode.js`/`PackIntegerNode.js`
 * /`UnpackIntegerNode.js` can be exercised and verified for correctness on a
 * real WebGPU backend that (in every environment this repo's tests actually
 * run in) *does* support the feature - there would otherwise be no way to
 * reach that code path at all. Pass `undefined` to restore normal detection.
 *
 * @param {boolean|undefined} value - The value to force, or `undefined` to
 * restore real detection.
 */
export function setPackedIntegerDotProductFeatureOverrideForTesting( value ) {

	testOverride = value;

}

export default supportsPackedIntegerDotProductFeature;

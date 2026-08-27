import { wgslFn } from 'three/tsl';

/**
 * Optional, opt-in inference-time MLP decoder path using WebGPU's
 * `packed_4x8_integer_dot_product` WGSL language feature (see
 * https://developer.mozilla.org/en-US/docs/Web/API/WGSLLanguageFeatures and
 * this repo's .cursor/plans/ntc_paper_gap_04_packed_int8_dot_product.plan.md
 * for the full investigation) - `dot4I8Packed`/`dot4U8Packed` compute a
 * 4-wide int8 dot product (4 signed-8-bit multiply-adds, accumulated as a
 * single i32) from two `u32`-packed operands in one call, which can let a
 * hidden layer's `dot(weightsRow, inputVector)` execute as one native
 * integer instruction instead of 4 scalar `f32` multiply-adds - potentially
 * faster than the existing `mat4 * vec4` fp32 path (NTCMLPTSL.js), at the
 * cost of int8 numerical precision on the *hidden* layers only (never the
 * final linear output layer - see NTCMLPTSL.js's evaluateHiddenLayerDot4I8
 * doc comment for why).
 *
 * This is a **language feature**, not a GPU/device feature: it's checked via
 * `navigator.gpu.wgslLanguageFeatures.has(...)` (a property of the `GPU`
 * object itself, available before ever requesting an adapter/device) rather
 * than `renderer.hasFeature(...)`/`device.features` (see
 * `supportsPackedDotProduct` below) - a real, if perhaps confusing,
 * distinction in the WebGPU spec between "does the WGSL compiler accept this
 * syntax" (language features) and "does the GPU expose this capability"
 * (device features, e.g. `shader-f16`, already used by NTCMLPTSL.js's
 * `supportsHalfPrecisionStorage`).
 *
 * That "language feature" distinction turns out to matter for more than just
 * which capability list to check: unlike a WGSL *extension* (e.g. `f16`,
 * `subgroups`), which requires an `enable <name>;` directive as the first
 * thing in the shader module before any of its syntax/builtins can be used, a
 * WGSL *language feature* needs no `enable` directive at all - the name only
 * shows up in `navigator.gpu.wgslLanguageFeatures` as a way to *detect*
 * support, not to opt into it. This was confirmed empirically, not just read
 * off the spec: an early version of this file tried emitting `enable
 * packed_4x8_integer_dot_product;` (mirroring how `shader-f16` needs
 * `enable f16;`) and Chromium's WGSL compiler rejected it outright with
 * "expected extension" - `dot4I8Packed`/`dot4U8Packed` are directly callable
 * the moment `supportsPackedDotProduct` returns true, no directive, no
 * special material-level wiring (e.g. no `NodeMaterial.setup(builder)`
 * override) required at all.
 */

/**
 * True when `renderer` can compile and run WGSL using
 * `packed_4x8_integer_dot_product` - WebGPU backend only, and only when the
 * browser's WGSL compiler actually advertises the language feature (see the
 * module doc comment above for why this checks `navigator.gpu.
 * wgslLanguageFeatures`, not `renderer.hasFeature(...)`). `renderer` may be
 * omitted or not yet `init()`-ed - this just means "not supported", never a
 * throw, matching NTCMLPTSL.js's `supportsHalfPrecisionStorage` convention.
 */
function supportsPackedDotProduct( renderer ) {

	return Boolean(
		renderer &&
		renderer.backend &&
		renderer.backend.isWebGPUBackend === true &&
		typeof navigator !== 'undefined' &&
		navigator.gpu &&
		navigator.gpu.wgslLanguageFeatures &&
		navigator.gpu.wgslLanguageFeatures.has( 'packed_4x8_integer_dot_product' )
	);

}

// Raw WGSL wrappers for the two builtins this feature provides. Written as
// trivial one-line pass-throughs (rather than calling the builtins directly
// from a larger hand-written WGSL block) so the *only* thing routed through
// raw WGSL is the builtin call itself - everything around it (packing,
// scaling, accumulation) stays in ordinary TSL, tested and readable the same
// way as the rest of this addon.
const dot4I8PackedFn = /*@__PURE__*/ wgslFn( `
fn ntc_dot4I8Packed( a: u32, b: u32 ) -> i32 {
	return dot4I8Packed( a, b );
}` );

const dot4U8PackedFn = /*@__PURE__*/ wgslFn( `
fn ntc_dot4U8Packed( a: u32, b: u32 ) -> u32 {
	return dot4U8Packed( a, b );
}` );

/**
 * TSL wrapper: 4-wide signed-int8 packed dot product. `aNode`/`bNode` must be
 * `uint`-typed TSL nodes, each holding 4 signed 8-bit lanes packed
 * little-endian (byte 0 = bits [0:8), ... byte 3 = bits [24:32) - see
 * `packInt8x4` below for the matching CPU-side packer). Returns a signed
 * `int` TSL node holding the exact integer sum of the 4 pairwise products
 * (no precision loss - int8*int8 accumulated in i32 can't overflow: the
 * largest possible term is 128*128=16384, times 4 terms is 65536, far inside
 * i32's range).
 */
function dot4I8PackedTSL( aNode, bNode ) {

	return dot4I8PackedFn( { a: aNode, b: bNode } );

}

/**
 * TSL wrapper: 4-wide unsigned-int8 packed dot product - see
 * `dot4I8PackedTSL` above, unsigned lanes instead of signed.
 */
function dot4U8PackedTSL( aNode, bNode ) {

	return dot4U8PackedFn( { a: aNode, b: bNode } );

}

/**
 * Packs up to 4 signed 8-bit integers (each expected in `[-128, 127]`,
 * clamped otherwise - callers should already have rounded/clamped via
 * `quantizeSymmetricInt8` below) into one little-endian-packed `u32`,
 * zero-padding past `values.length` - the CPU-side mirror of what
 * `dot4I8PackedTSL` expects on the GPU. Returned as an *unsigned* 32-bit
 * integer (`>>> 0`) since that's the bit pattern a `u32` uniform/storage
 * value needs, even though the 4 individual bytes are interpreted as signed
 * by `dot4I8Packed` itself.
 */
function packInt8x4( values ) {

	let packed = 0;

	for ( let i = 0; i < 4; i ++ ) {

		const value = Math.max( - 128, Math.min( 127, Math.round( values[ i ] || 0 ) ) );
		packed |= ( value & 0xFF ) << ( 8 * i );

	}

	return packed >>> 0;

}

/**
 * Packs up to 4 *unsigned* 8-bit integers (each expected in `[0, 255]`,
 * clamped otherwise) into one little-endian-packed `u32`, zero-padding past
 * `values.length` - the CPU-side mirror of what `dot4U8PackedTSL` expects.
 * Unlike `packInt8x4`, no sign/two's-complement handling is needed: an
 * unsigned byte's bit pattern already *is* its value, so `value & 0xFF` alone
 * (no prior clamp-then-reinterpret step) is enough. Not currently used by
 * this addon's own MLP evaluator (`NTCMLPTSL.js`'s `evaluateHiddenLayerDot4I8`
 * only needs the signed path, since weights/activations are quantized
 * symmetrically - see `quantizeSymmetricInt8`), but exported alongside
 * `dot4U8PackedTSL` so that primitive has a real, testable CPU-side
 * counterpart rather than being unusable dead weight.
 */
function packUint8x4( values ) {

	let packed = 0;

	for ( let i = 0; i < 4; i ++ ) {

		const value = Math.max( 0, Math.min( 255, Math.round( values[ i ] || 0 ) ) );
		packed |= ( value & 0xFF ) << ( 8 * i );

	}

	return packed >>> 0;

}

/**
 * Quantizes a single value to a signed int8 level given a symmetric scale
 * (`bound / 127`, see `computeSymmetricScale` below) - `round(x / scale)`,
 * clamped to `[-128, 127]`. Used identically for both weights and
 * activations, since both use the same symmetric (zero-centered, no
 * zero-point offset) quantization scheme - simpler than NTCQuantization.js's
 * latent-grid `zeroPreserving` asymmetric scheme, appropriate here since
 * weights are already roughly zero-centered (He initialization) and
 * activation bounds are derived symmetrically (see
 * `computeActivationBounds` below), so there's no equivalent asymmetry to
 * correct for.
 */
function quantizeSymmetricInt8( value, scale ) {

	if ( scale === 0 ) return 0;

	return Math.max( - 128, Math.min( 127, Math.round( value / scale ) ) );

}

/**
 * The symmetric quantization scale for a value bounded by `+/-bound`
 * (already guaranteed to cover the true range - see
 * `computeActivationBounds`/`computeSymmetricWeightScale` below) - the
 * largest step size that still keeps every representable value within the
 * 8-bit signed range `[-128, 127]`. `bound <= 0` (a degenerate all-zero
 * case) returns a scale of 1 rather than dividing by zero - quantizing an
 * all-zero tensor to all-zero int8 values either way.
 */
function computeSymmetricScale( bound ) {

	return bound > 0 ? bound / 127 : 1;

}

export {
	supportsPackedDotProduct,
	dot4I8PackedTSL,
	dot4U8PackedTSL,
	packInt8x4,
	packUint8x4,
	quantizeSymmetricInt8,
	computeSymmetricScale
};

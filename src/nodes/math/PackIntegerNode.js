import { ivec4, nodeProxyIntent, uint, uvec4 } from '../tsl/TSLCore.js';
import MathNode from './MathNode.js';
import supportsPackedIntegerDotProductFeature from './PackedIntegerFeatureDetection.js';

/**
 * This node represents an operation that packs 4 8-bit integers of a vector
 * into a single unsigned 32-bit integer - backed by WGSL's `pack4xI8`/
 * `pack4xU8` builtins (the same "Packed 4x8 Integer Dot Product" WGSL
 * language feature that {@link PackedDotProductNode} wraps - see
 * https://developer.mozilla.org/en-US/docs/Web/API/WGSLLanguageFeatures) on
 * WebGPU, with a hand-written GLSL polyfill so the same TSL call also works
 * on the WebGL fallback backend.
 *
 * Like `dot4I8Packed`/`dot4U8Packed`, this is a WGSL *language feature*, not
 * an *extension* - no `enable` directive is required, confirmed empirically
 * the same way (see `PackedDotProductNode.js`'s doc comment for the full
 * investigation).
 *
 * Complements the existing `packSnorm4x8`/`packUnorm4x8` (`PackFloatNode.js`)
 * pair, which pack *normalized floats* into 8-bit lanes; this node packs
 * *integers already in `[-128, 127]`/`[0, 255]`* directly, with no float
 * normalization step - useful when the source values are already quantized
 * integers (e.g. this repo's NTC addon's int8 MLP evaluator), where an extra
 * round-trip through a normalized float would be wasted work.
 *
 * The WGSL builtins are a *language feature*, not a device capability every
 * WebGPU implementation is guaranteed to support (see
 * `PackedIntegerFeatureDetection.js`) - `setup()` below checks for that and
 * falls back to a hand-written TSL polyfill (built from ordinary bitwise
 * operators) when running on WebGPU without it, rather than assuming WebGPU
 * always implies support the way earlier versions of this node did.
 *
 * @augments MathNode
 */
class PackIntegerNode extends MathNode {

	static get type() {

		return 'PackIntegerNode';

	}

	/**
	 * Constructs a new pack integer node.
	 *
	 * @param {'pack4xI8'|'pack4xU8'} method - The method name.
	 * @param {Node<ivec4|uvec4>} aNode - The 4-component integer vector to pack.
	 */
	constructor( method, aNode ) {

		super( method, aNode );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isPackIntegerNode = true;

	}

	generateNodeType() {

		return 'uint';

	}

	/**
	 * On WebGPU without the "Packed 4x8 Integer Dot Product" language
	 * feature, falls back to the hand-written TSL polyfill below rather than
	 * emitting a call to a builtin that may not exist - see this class's own
	 * doc comment. Otherwise defers to `MathNode.setup()`/this class's own
	 * `generate()`, unchanged (native call on WebGPU with the feature, GLSL
	 * polyfill via `builder.getMethod()` on the WebGL fallback backend).
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {?Node} The output node, or `null` to defer to `generate()`.
	 */
	setup( builder ) {

		const { renderer } = builder;

		if ( renderer.backend.isWebGPUBackend && ! supportsPackedIntegerDotProductFeature() ) {

			return this.method === PackIntegerNode.PACK4X_I8 ?
				pack4xI8Fallback( this.aNode ) :
				pack4xU8Fallback( this.aNode );

		}

		return super.setup( builder );

	}

	generate( builder, output ) {

		const properties = builder.getNodeProperties( this );

		if ( properties.outputNode ) {

			// `setup()` above already built a fallback subgraph (WebGPU
			// without the language feature) - build that instead of the
			// native call below. `MathNode.generate()` does this same check
			// for its own special-cased methods; this class's own
			// `generate()` override needs its own copy since it otherwise
			// bypasses `MathNode.generate()` entirely.
			return super.generate( builder, output );

		}

		const { method, aNode } = this;

		const inputType = method === PackIntegerNode.PACK4X_I8 ? 'ivec4' : 'uvec4';
		const type = this.getNodeType( builder );
		const a = aNode.build( builder, inputType );

		const nativeMethod = builder.getMethod( method );

		return builder.format( `${ nativeMethod }( ${ a } )`, type, output );

	}

}

PackIntegerNode.PACK4X_I8 = 'pack4xI8';
PackIntegerNode.PACK4X_U8 = 'pack4xU8';

export default PackIntegerNode;

/**
 * Hand-written TSL polyfill for `pack4xI8`, used by this node's own
 * `setup()` on WebGPU without the language feature. Built entirely from
 * ordinary bitwise operators, which are core WGSL/GLSL functionality with no
 * capability check of their own needed - masking each lane to its low 8
 * bits (`bitAnd( 0xFF )`, which yields the same little-endian
 * two's-complement byte a native `pack4xI8` would for a value already
 * clamped to `[-128, 127]`) and shifting it into place.
 *
 * @param {Node<ivec4>} aNode - The 4-component signed integer vector to pack.
 * @returns {Node<uint>}
 */
export function pack4xI8Fallback( aNode ) {

	const v = ivec4( aNode );

	const bx = uint( v.x.bitAnd( 0xFF ) );
	const by = uint( v.y.bitAnd( 0xFF ) );
	const bz = uint( v.z.bitAnd( 0xFF ) );
	const bw = uint( v.w.bitAnd( 0xFF ) );

	return bx.bitOr( by.shiftLeft( 8 ) ).bitOr( bz.shiftLeft( 16 ) ).bitOr( bw.shiftLeft( 24 ) );

}

/**
 * Hand-written TSL polyfill for `pack4xU8` - see {@link pack4xI8Fallback},
 * unsigned lanes instead of signed.
 *
 * @param {Node<uvec4>} aNode - The 4-component unsigned integer vector to pack.
 * @returns {Node<uint>}
 */
export function pack4xU8Fallback( aNode ) {

	const v = uvec4( aNode );

	const bx = v.x.bitAnd( 0xFF );
	const by = v.y.bitAnd( 0xFF );
	const bz = v.z.bitAnd( 0xFF );
	const bw = v.w.bitAnd( 0xFF );

	return bx.bitOr( by.shiftLeft( 8 ) ).bitOr( bz.shiftLeft( 16 ) ).bitOr( bw.shiftLeft( 24 ) );

}

/**
 * Packs a 4-component vector of signed 8-bit integers (each expected in
 * `[-128, 127]`) into a single `uint`, little-endian (component `x` occupies
 * bits `[0, 8)`, ... `w` occupies bits `[24, 32)`).
 *
 * Native on WebGPU (WGSL's `pack4xI8`); emulated with a small GLSL polyfill
 * on the WebGL fallback backend.
 *
 * @tsl
 * @function
 * @param {Node<ivec4>} a - The 4-component signed integer vector to pack.
 * @returns {Node<uint>}
 */
export const pack4xI8 = /*@__PURE__*/ nodeProxyIntent( PackIntegerNode, PackIntegerNode.PACK4X_I8 ).setParameterLength( 1 );

/**
 * Packs a 4-component vector of unsigned 8-bit integers (each expected in
 * `[0, 255]`) into a single `uint` - see {@link pack4xI8}, unsigned lanes
 * instead of signed.
 *
 * Native on WebGPU (WGSL's `pack4xU8`); emulated with a small GLSL polyfill
 * on the WebGL fallback backend.
 *
 * @tsl
 * @function
 * @param {Node<uvec4>} a - The 4-component unsigned integer vector to pack.
 * @returns {Node<uint>}
 */
export const pack4xU8 = /*@__PURE__*/ nodeProxyIntent( PackIntegerNode, PackIntegerNode.PACK4X_U8 ).setParameterLength( 1 );

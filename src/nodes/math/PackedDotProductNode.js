import { nodeProxyIntent } from '../tsl/TSLCore.js';
import MathNode from './MathNode.js';
import supportsPackedIntegerDotProductFeature from './PackedIntegerFeatureDetection.js';
import { unpack4xI8Fallback, unpack4xU8Fallback } from './UnpackIntegerNode.js';

/**
 * This node represents a 4-wide dot product of two operands, each packed as
 * 4 little-endian 8-bit integer lanes into a single `uint` - backed by
 * WGSL's `dot4I8Packed`/`dot4U8Packed` builtins (the "Packed 4x8 Integer Dot
 * Product" WGSL language feature - see
 * https://developer.mozilla.org/en-US/docs/Web/API/WGSLLanguageFeatures) on
 * WebGPU, with a hand-written GLSL polyfill so the same TSL call also works
 * on the WebGL fallback backend.
 *
 * Unlike a WGSL *extension* (e.g. `f16`, `subgroups`), this WGSL *language
 * feature* needs no `enable` directive - it's simply always callable
 * *once supported* - so (unlike some other WebGPU-only capabilities in this
 * codebase) this node needs no capability check to know *which directive* to
 * emit. It does, however, still need one to know whether the builtin exists
 * at all: this is an optional WGSL language feature, not something every
 * WebGPU implementation is guaranteed to support (see
 * `PackedIntegerFeatureDetection.js`) - `setup()` below checks for that and
 * falls back to a hand-written TSL polyfill (built by unpacking both
 * operands via `UnpackIntegerNode.js`'s own fallback and multiplying
 * component-wise) when running on WebGPU without it, rather than assuming
 * WebGPU always implies support the way an earlier version of this node
 * did. Callers that need to know whether the *native* (rather than
 * polyfilled) path is in use can check `renderer.backend.isWebGPUBackend`
 * together with `supportsPackedIntegerDotProductFeature()` directly.
 *
 * @augments MathNode
 */
class PackedDotProductNode extends MathNode {

	static get type() {

		return 'PackedDotProductNode';

	}

	/**
	 * Constructs a new packed dot product node.
	 *
	 * @param {'dot4I8Packed'|'dot4U8Packed'} method - The method name.
	 * @param {Node<uint>} aNode - The first packed operand.
	 * @param {Node<uint>} bNode - The second packed operand.
	 */
	constructor( method, aNode, bNode ) {

		super( method, aNode, bNode );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isPackedDotProductNode = true;

	}

	/**
	 * The signed variant (`dot4I8Packed`) returns a signed `int` even though
	 * both operands are `uint` - unlike `MathNode`'s own generic methods
	 * (whose output type always matches the input type unless the method is
	 * one of a handful hardcoded exceptions - `dot`, `cross`, etc.), so this
	 * is overridden rather than relying on `MathNode.generateNodeType`. The
	 * unsigned variant (`dot4U8Packed`) returns `uint`, matching its inputs.
	 *
	 * @return {string} The node type.
	 */
	generateNodeType() {

		return this.method === PackedDotProductNode.DOT4_I8_PACKED ? 'int' : 'uint';

	}

	/**
	 * On WebGPU without the "Packed 4x8 Integer Dot Product" language
	 * feature, falls back to unpacking both operands (via
	 * `UnpackIntegerNode.js`'s own fallback) and computing the dot product
	 * manually, rather than emitting a call to a builtin that may not exist -
	 * see this class's own doc comment. Otherwise defers to
	 * `MathNode.setup()`/this class's own `generate()`, unchanged (native
	 * call on WebGPU with the feature, GLSL polyfill via
	 * `builder.getMethod()` on the WebGL fallback backend).
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {?Node} The output node, or `null` to defer to `generate()`.
	 */
	setup( builder ) {

		const { renderer } = builder;

		if ( renderer.backend.isWebGPUBackend && ! supportsPackedIntegerDotProductFeature() ) {

			const unpackFn = this.method === PackedDotProductNode.DOT4_I8_PACKED ? unpack4xI8Fallback : unpack4xU8Fallback;

			const av = unpackFn( this.aNode );
			const bv = unpackFn( this.bNode );
			const p = av.mul( bv );

			return p.x.add( p.y ).add( p.z ).add( p.w );

		}

		return super.setup( builder );

	}

	generate( builder, output ) {

		const properties = builder.getNodeProperties( this );

		if ( properties.outputNode ) {

			// `setup()` above already built a fallback subgraph (WebGPU
			// without the language feature) - build that instead of the
			// native call below. See PackIntegerNode.js's own copy of this
			// same check for why it's needed here.
			return super.generate( builder, output );

		}

		const { method, aNode, bNode } = this;

		const type = this.getNodeType( builder );
		const a = aNode.build( builder, 'uint' );
		const b = bNode.build( builder, 'uint' );

		const nativeMethod = builder.getMethod( method );

		return builder.format( `${ nativeMethod }( ${ a }, ${ b } )`, type, output );

	}

}

PackedDotProductNode.DOT4_I8_PACKED = 'dot4I8Packed';
PackedDotProductNode.DOT4_U8_PACKED = 'dot4U8Packed';

export default PackedDotProductNode;

/**
 * Computes a 4-wide packed *signed* 8-bit integer dot product: interprets
 * `a` and `b` as 4 little-endian-packed signed 8-bit lanes each (byte 0 =
 * bits `[0, 8)`, ... byte 3 = bits `[24, 32)`), and returns the exact
 * integer sum of the 4 pairwise products (`int8 * int8` accumulated in
 * `i32` - the largest possible term is `128 * 128 = 16384`, times 4 terms is
 * `65536`, far inside `i32`'s range, so this can never overflow).
 *
 * Native on WebGPU (WGSL's `dot4I8Packed`); emulated with a small GLSL
 * polyfill on the WebGL fallback backend.
 *
 * @tsl
 * @function
 * @param {Node<uint> | number} a - The first packed operand.
 * @param {Node<uint> | number} b - The second packed operand.
 * @returns {Node<int>}
 */
export const dot4I8Packed = /*@__PURE__*/ nodeProxyIntent( PackedDotProductNode, PackedDotProductNode.DOT4_I8_PACKED ).setParameterLength( 2 );

/**
 * Computes a 4-wide packed *unsigned* 8-bit integer dot product - see
 * {@link dot4I8Packed}, unsigned lanes instead of signed (no overflow risk
 * either: `255 * 255 = 65025`, times 4 terms is `260100`, still far inside
 * `u32`'s range).
 *
 * Native on WebGPU (WGSL's `dot4U8Packed`); emulated with a small GLSL
 * polyfill on the WebGL fallback backend.
 *
 * @tsl
 * @function
 * @param {Node<uint> | number} a - The first packed operand.
 * @param {Node<uint> | number} b - The second packed operand.
 * @returns {Node<uint>}
 */
export const dot4U8Packed = /*@__PURE__*/ nodeProxyIntent( PackedDotProductNode, PackedDotProductNode.DOT4_U8_PACKED ).setParameterLength( 2 );

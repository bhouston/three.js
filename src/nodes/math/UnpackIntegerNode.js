import { int, ivec4, nodeProxyIntent, uint, uvec4 } from '../tsl/TSLCore.js';
import MathNode from './MathNode.js';
import supportsPackedIntegerDotProductFeature from './PackedIntegerFeatureDetection.js';

/**
 * This node represents an operation that unpacks a single unsigned 32-bit
 * integer into 4 8-bit integer lanes - backed by WGSL's `unpack4xI8`/
 * `unpack4xU8` builtins, the inverse of {@link PackIntegerNode}'s
 * `pack4xI8`/`pack4xU8` (both part of the "Packed 4x8 Integer Dot Product"
 * WGSL language feature - see
 * https://developer.mozilla.org/en-US/docs/Web/API/WGSLLanguageFeatures) on
 * WebGPU, with a hand-written GLSL polyfill so the same TSL call also works
 * on the WebGL fallback backend.
 *
 * Complements the existing `unpackSnorm4x8`/`unpackUnorm4x8`
 * (`UnpackFloatNode.js`) pair, which unpack 8-bit lanes into *normalized
 * floats*; this node unpacks the lanes as plain integers, with no float
 * normalization step.
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
class UnpackIntegerNode extends MathNode {

	static get type() {

		return 'UnpackIntegerNode';

	}

	/**
	 * Constructs a new unpack integer node.
	 *
	 * @param {'unpack4xI8'|'unpack4xU8'} method - The method name.
	 * @param {Node<uint>} aNode - The packed unsigned integer to unpack.
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
		this.isUnpackIntegerNode = true;

	}

	/**
	 * The signed variant (`unpack4xI8`) returns a signed `ivec4`; the unsigned
	 * variant (`unpack4xU8`) returns `uvec4`.
	 *
	 * @return {string} The node type.
	 */
	generateNodeType() {

		return this.method === UnpackIntegerNode.UNPACK4X_I8 ? 'ivec4' : 'uvec4';

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

			return this.method === UnpackIntegerNode.UNPACK4X_I8 ?
				unpack4xI8Fallback( this.aNode ) :
				unpack4xU8Fallback( this.aNode );

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

		const { method, aNode } = this;

		const type = this.getNodeType( builder );
		const a = aNode.build( builder, 'uint' );

		const nativeMethod = builder.getMethod( method );

		return builder.format( `${ nativeMethod }( ${ a } )`, type, output );

	}

}

UnpackIntegerNode.UNPACK4X_I8 = 'unpack4xI8';
UnpackIntegerNode.UNPACK4X_U8 = 'unpack4xU8';

export default UnpackIntegerNode;

/**
 * Hand-written TSL polyfill for `unpack4xI8`, used by this node's own
 * `setup()` on WebGPU without the language feature, and reused by
 * `PackedDotProductNode.js`'s own fallback for `dot4I8Packed` (which needs
 * to unpack both operands before it can compute the dot product manually).
 * Built entirely from ordinary bitwise operators/`bitcast`, which are core
 * WGSL/GLSL functionality with no capability check of their own needed.
 *
 * Sign-extends each byte via the classic XOR-bias trick (`(byte ^ 128) -
 * 128`, for `byte` already masked to `[0, 255]`) rather than this feature's
 * own GLSL polyfill's shift-left-then-arithmetic-shift-right idiom
 * (`GLSLNodeBuilder.js`): shifting a lane up to the top of the word
 * overflows a `u32` at *compile time* in WGSL when `aNode` happens to be a
 * constant (`'<value> << 24' cannot be represented as 'u32'`, confirmed
 * directly against Chromium's WGSL compiler) - runtime overflow is fine and
 * well-defined, but WGSL rejects the *constant-expression* form outright,
 * unlike GLSL, which has no equivalent restriction. The XOR-bias trick
 * never produces an intermediate outside `[-128, 255]`, so it can't trip
 * this either way.
 *
 * @param {Node<uint> | number} aNode - The packed integer to unpack.
 * @returns {Node<ivec4>}
 */
export function unpack4xI8Fallback( aNode ) {

	const a = uint( aNode );

	const lane = ( shiftRightAmount ) => {

		const byte = int( a.shiftRight( shiftRightAmount ).bitAnd( 0xFF ) );
		return byte.bitXor( 128 ).sub( 128 );

	};

	return ivec4( lane( 0 ), lane( 8 ), lane( 16 ), lane( 24 ) );

}

/**
 * Hand-written TSL polyfill for `unpack4xU8` - see
 * {@link unpack4xI8Fallback}, unsigned lanes instead of signed (no sign
 * extension needed, just a mask per lane).
 *
 * @param {Node<uint> | number} aNode - The packed integer to unpack.
 * @returns {Node<uvec4>}
 */
export function unpack4xU8Fallback( aNode ) {

	const a = uint( aNode );

	return uvec4(
		a.bitAnd( 0xFF ),
		a.shiftRight( 8 ).bitAnd( 0xFF ),
		a.shiftRight( 16 ).bitAnd( 0xFF ),
		a.shiftRight( 24 ).bitAnd( 0xFF )
	);

}

/**
 * Unpacks a `uint` into a 4-component vector of signed 8-bit integers,
 * little-endian (bits `[0, 8)` become component `x`, ... bits `[24, 32)`
 * become `w`), each sign-extended to a full `int`.
 *
 * Native on WebGPU (WGSL's `unpack4xI8`); emulated with a small GLSL polyfill
 * on the WebGL fallback backend.
 *
 * @tsl
 * @function
 * @param {Node<uint> | number} a - The packed integer to unpack.
 * @returns {Node<ivec4>}
 */
export const unpack4xI8 = /*@__PURE__*/ nodeProxyIntent( UnpackIntegerNode, UnpackIntegerNode.UNPACK4X_I8 ).setParameterLength( 1 );

/**
 * Unpacks a `uint` into a 4-component vector of unsigned 8-bit integers -
 * see {@link unpack4xI8}, unsigned lanes instead of signed.
 *
 * Native on WebGPU (WGSL's `unpack4xU8`); emulated with a small GLSL polyfill
 * on the WebGL fallback backend.
 *
 * @tsl
 * @function
 * @param {Node<uint> | number} a - The packed integer to unpack.
 * @returns {Node<uvec4>}
 */
export const unpack4xU8 = /*@__PURE__*/ nodeProxyIntent( UnpackIntegerNode, UnpackIntegerNode.UNPACK4X_U8 ).setParameterLength( 1 );

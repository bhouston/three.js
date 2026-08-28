import { nodeProxyIntent } from '../tsl/TSLCore.js';
import MathNode from './MathNode.js';

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

	generate( builder, output ) {

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

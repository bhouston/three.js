import { nodeProxyIntent } from '../tsl/TSLCore.js';
import MathNode from './MathNode.js';

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

	generate( builder, output ) {

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

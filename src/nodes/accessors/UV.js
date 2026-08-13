import Node from '../core/Node.js';
import { attribute } from '../core/AttributeNode.js';
import { context } from '../core/ContextNode.js';
import { hashString } from '../core/NodeUtils.js';

/**
 * A context-aware accessor for an indexed UV geometry attribute.
 *
 * By default, this node resolves to {@link uv}. A `getUVAttribute` builder
 * context callback can replace the source coordinate without replacing
 * concrete attribute-node instances.
 *
 * @augments Node
 */
class SemanticUVNode extends Node {

	static get type() {

		return 'SemanticUVNode';

	}

	/**
	 * Constructs a new semantic UV node.
	 *
	 * @param {number} [index=0] - The uv index.
	 */
	constructor( index = 0 ) {

		super( 'vec2' );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isSemanticUVNode = true;

		/**
		 * The UV attribute index.
		 *
		 * @type {number}
		 * @default 0
		 */
		this.index = index;

	}

	customCacheKey() {

		return hashString( `${this.type}:${this.index}` );

	}

	setup( builder ) {

		const sourceNode = uv( this.index );
		const getUVAttribute = builder.context.getUVAttribute;

		if ( getUVAttribute === null || getUVAttribute === undefined ) {

			return sourceNode;

		}

		const overrideNode = getUVAttribute( this.index, this, builder );

		if ( overrideNode === null || overrideNode === undefined ) {

			return sourceNode;

		}

		// Disable this callback while building the replacement so a replacement
		// that references semanticUV() falls back to its geometry attribute.
		return context( overrideNode, { getUVAttribute: null } );

	}

	serialize( data ) {

		super.serialize( data );

		data.index = this.index;

	}

	deserialize( data ) {

		super.deserialize( data );

		this.index = data.index;

	}

}

export default SemanticUVNode;

/**
 * TSL function for creating an uv attribute node with the given index.
 *
 * @tsl
 * @function
 * @param {number} [index=0] - The uv index.
 * @return {AttributeNode<vec2>} The uv attribute node.
 */
export const uv = ( index = 0 ) => attribute( 'uv' + ( index > 0 ? index : '' ), 'vec2' );

/**
 * TSL function for creating a context-aware indexed UV geometry accessor.
 *
 * A `getUVAttribute( index, sourceNode, builder )` builder-context callback can
 * override the returned source-space coordinate. Without that callback this
 * accessor resolves to {@link uv}.
 *
 * @tsl
 * @function
 * @param {number} [index=0] - The uv index.
 * @return {SemanticUVNode} The semantic uv node.
 */
export const semanticUV = ( index = 0 ) => new SemanticUVNode( index );

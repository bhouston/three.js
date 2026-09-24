import LightingNode from './LightingNode.js';

/**
 * A generic class that can be used by nodes which provide the
 * indirect specular radiance of the scene. E.g. screen space
 * reflections can be used as input for this module. Used in {@link NodeMaterial}.
 *
 * The radiance replaces the environment map radiance instead of adding to it, so the
 * node should provide the complete radiance, falling back to the environment where it
 * has no information of its own (see FidelityFX SSSR).
 *
 * @augments LightingNode
 */
class RadianceNode extends LightingNode {

	static get type() {

		return 'RadianceNode';

	}

	/**
	 * Constructs a new radiance node.
	 *
	 * @param {Node<vec3>} node - A node providing the indirect specular radiance.
	 */
	constructor( node ) {

		super();

		/**
		 * A node providing the indirect specular radiance.
		 *
		 * @type {Node<vec3>}
		 */
		this.node = node;

	}

	setup( builder ) {

		// NodeMaterial creates this node after the environment node and the lighting nodes are
		// sorted by id, so this replaces the radiance the environment has already added.

		builder.context.radiance.assign( this.node );

	}

}

export default RadianceNode;

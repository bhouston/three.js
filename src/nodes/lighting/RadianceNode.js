import LightingNode from './LightingNode.js';

/**
 * A generic class that can be used by nodes which contribute
 * indirect specular radiance to the scene. E.g. screen space
 * reflections can be used as input for this module. Used in {@link NodeMaterial}.
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
	 * @param {Node<vec3>} node - A node contributing indirect specular radiance.
	 */
	constructor( node ) {

		super();

		/**
		 * A node contributing indirect specular radiance.
		 *
		 * @type {Node<vec3>}
		 */
		this.node = node;

	}

	setup( builder ) {

		builder.context.radiance.addAssign( this.node );

	}

}

export default RadianceNode;

import { LightingNode } from 'three/webgpu';
import { float, vec3, vec4 } from 'three/tsl';

/**
 * A source is a `vec4` node, or a function of the node builder that returns one or `null`.
 * RGB is the incoming light, not premultiplied. Alpha is the coverage, the part of the
 * pixel the source answers.
 *
 * @typedef {(Node<vec4>|function(NodeBuilder): ?Node<vec4>)} CoverageSource
 */

/**
 * Resolves ordered lists of indirect light sources by coverage: each source fills what
 * the sources before it left open, and the default lighting of the material fills the rest.
 *
 * Diffuse sources return irradiance, specular sources the radiance of the base lobe.
 *
 * @augments LightingNode
 * @three_import import { coverageIndirect } from 'three/addons/tsl/lighting/CoverageLightingNode.js';
 */
class CoverageLightingNode extends LightingNode {

	static get type() {

		return 'CoverageLightingNode';

	}

	/**
	 * Constructs a new coverage lighting node.
	 *
	 * @param {Array<LightingNode>} lightings - The default lighting nodes of the material, the last fallback.
	 * @param {Array<CoverageSource>} [diffuse=[]] - The indirect diffuse sources, in order.
	 * @param {Array<CoverageSource>} [specular=[]] - The indirect specular sources, in order.
	 */
	constructor( lightings, diffuse = [], specular = [] ) {

		super();

		this.lightings = lightings;
		this.diffuse = diffuse;
		this.specular = specular;

	}

	setup( builder ) {

		for ( const lighting of this.lightings ) lighting.build( builder );

		const { radiance, irradiance, iblIrradiance } = builder.context;

		// ponytail: the lighting model still occludes both slots as a whole, per-source occlusion when a source needs to opt out

		const diffuse = resolve( this.diffuse, builder );

		if ( diffuse !== null ) {

			irradiance.assign( diffuse.light.add( irradiance.mul( diffuse.remaining ) ) );
			iblIrradiance.mulAssign( diffuse.remaining );

		}

		const specular = resolve( this.specular, builder );

		if ( specular !== null ) {

			radiance.assign( specular.light.add( radiance.mul( specular.remaining ) ) );

		}

	}

}

export default CoverageLightingNode;

function resolve( sources, builder ) {

	let light = null, remaining = null;

	for ( const source of sources ) {

		const node = typeof source === 'function' ? source( builder ) : source;

		if ( node === null ) continue;

		const value = vec4( node ).toVar();
		const weight = remaining === null ? value.a : remaining.mul( value.a );

		light = light === null ? value.rgb.mul( weight ) : light.add( value.rgb.mul( weight ) );
		remaining = ( remaining === null ? float( 1 ) : remaining ).sub( weight ).toVar();

	}

	return light === null ? null : { light: vec3( light ), remaining };

}

/**
 * Returns a callback for `builtinIndirectContext()` that resolves the given sources by coverage
 * in front of the default indirect lighting of each material. Transparent materials keep their
 * default lighting.
 *
 * @tsl
 * @function
 * @param {Object} sources - The source lists.
 * @param {Array<CoverageSource>} [sources.diffuse] - The indirect diffuse sources, in order.
 * @param {Array<CoverageSource>} [sources.specular] - The indirect specular sources, in order.
 * @returns {function(Array<LightingNode>, NodeBuilder): Array<LightingNode>}
 */
export const coverageIndirect = ( { diffuse, specular } ) => ( lightings, { material } ) => {

	if ( material.transparent === true ) return lightings;

	return [ new CoverageLightingNode( lightings, diffuse, specular ) ];

};

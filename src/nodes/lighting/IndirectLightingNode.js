import LightingNode from './LightingNode.js';
import { roughness, clearcoatRoughness, retroreflectivity } from '../core/PropertyNode.js';
import { cameraWorldMatrix } from '../accessors/Camera.js';
import { normalView, clearcoatNormalView, normalWorld } from '../accessors/Normal.js';
import { positionViewDirection } from '../accessors/Position.js';
import { bentNormalView } from '../accessors/AccessorsUtils.js';
import { vec4, pow4 } from '../tsl/TSLBase.js';

/**
 * A source of indirect light. It answers a query with a `vec4`: RGB is the light, premultiplied
 * by the coverage, and alpha is the coverage, the part of the query the source answers.
 * A source with a coverage of zero adds its light to the sources behind it.
 *
 * An indirect specular source returns radiance. Its query is the direction to look in, in world
 * space, and the roughness of the specular lobe. An indirect diffuse source returns irradiance.
 * Its query is the normal of the surface, in world space.
 *
 * A source is a function of the query and the builder, which can return `null` to sit a material
 * out. A source that holds one answer per pixel, such as a screen-space pass, can be a `vec4`
 * node. For indirect specular, such a node only answers the base lobe.
 *
 * @typedef {(Node<vec4>|function(Object, NodeBuilder): ?Node<vec4>)} IndirectLightSource
 */

/**
 * Computes the indirect light of a physically based material: the radiance of each of its
 * specular lobes, which is its indirect specular light, and the irradiance of its surface,
 * which is its indirect diffuse light.
 *
 * Each of the two is resolved from an ordered list of sources. A source fills what the sources
 * before it left open, and the environment fills the rest.
 *
 * @augments LightingNode
 */
class IndirectLightingNode extends LightingNode {

	static get type() {

		return 'IndirectLightingNode';

	}

	/**
	 * Constructs a new indirect lighting node.
	 *
	 * @param {Array<IndirectLightSource>} [specular=[]] - The indirect specular sources, in order.
	 * @param {Array<IndirectLightSource>} [diffuse=[]] - The indirect diffuse sources, in order.
	 * @param {?EnvironmentNode} [environmentNode=null] - The environment, the last source of both.
	 */
	constructor( specular = [], diffuse = [], environmentNode = null ) {

		super();

		this.specular = specular;
		this.diffuse = diffuse;
		this.environmentNode = environmentNode;

	}

	setup( builder ) {

		const { context, material } = builder;
		const { environmentNode } = this;

		if ( environmentNode !== null ) environmentNode.build( builder );

		// indirect diffuse

		const diffuse = this.resolve( this.diffuse, { normal: normalWorld }, true, builder );

		if ( diffuse.light !== null ) context.irradiance.addAssign( diffuse.light );

		if ( environmentNode !== null ) {

			// the environment is the last indirect diffuse source

			const irradiance = environmentNode.getIndirectDiffuse( { normal: normalWorld } );

			context.irradiance.addAssign( diffuse.remaining !== null ? irradiance.mul( diffuse.remaining ) : irradiance );

		}

		// indirect specular, one query per lobe

		const useAnisotropy = material.useAnisotropy === true || material.anisotropy > 0;
		const baseNormalView = useAnisotropy ? bentNormalView : normalView;

		let radiance = this.getRadiance( reflect( baseNormalView, roughness ), roughness, true, builder );

		if ( material.useRetroreflection === true || material.retroreflectivity > 0 ) {

			// the retroreflective lobe returns light toward its source, so it looks along the view direction

			const retroRadiance = this.getRadiance( toWorld( positionViewDirection, baseNormalView, roughness ), roughness, false, builder );

			if ( radiance !== null && retroRadiance !== null ) radiance = retroreflectivity.clamp().mix( radiance, retroRadiance );

		}

		if ( radiance !== null ) context.radiance.addAssign( radiance );

		const clearcoatRadiance = context.lightingModel.clearcoatRadiance;

		if ( clearcoatRadiance ) {

			const radiance = this.getRadiance( reflect( clearcoatNormalView, clearcoatRoughness ), clearcoatRoughness, false, builder );

			if ( radiance !== null ) clearcoatRadiance.addAssign( radiance );

		}

	}

	/**
	 * Returns the radiance of one specular lobe.
	 *
	 * @param {Node<vec3>} direction - The direction the lobe looks in, in world space.
	 * @param {Node<float>} roughness - The roughness of the lobe.
	 * @param {boolean} isBaseLobe - Whether this is the lobe that per-pixel sources hold the answer for.
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {?Node<vec3>} The radiance, `null` if nothing answers.
	 */
	getRadiance( direction, roughness, isBaseLobe, builder ) {

		const query = { direction, roughness };
		const { light, remaining } = this.resolve( this.specular, query, isBaseLobe, builder );

		if ( this.environmentNode === null ) return light;

		const environment = this.environmentNode.getIndirectSpecular( query );

		return light !== null ? light.add( environment.mul( remaining ) ) : environment;

	}

	/**
	 * Resolves a list of sources front to back.
	 *
	 * @param {Array<IndirectLightSource>} sources - The sources, in order.
	 * @param {Object} query - The query.
	 * @param {boolean} perPixel - Whether the sources that are nodes can answer the query.
	 * @param {NodeBuilder} builder - The current node builder.
	 * @return {{light: ?Node<vec3>, remaining: ?Node<float>}} The light of the sources and the coverage they left open, `null` if no source answers.
	 */
	resolve( sources, query, perPixel, builder ) {

		let light = null, remaining = null;

		for ( const source of sources ) {

			let value = null;

			if ( typeof source === 'function' ) value = source( query, builder );
			else if ( perPixel ) value = source;

			if ( value === null ) continue;

			value = vec4( value ).toVar();

			light = light === null ? value.rgb : light.add( value.rgb.mul( remaining ) );
			remaining = ( remaining === null ? value.a.oneMinus() : remaining.mul( value.a.oneMinus() ) ).toVar();

		}

		return { light, remaining };

	}

}

export default IndirectLightingNode;

// Mixing the direction with the normal is more accurate and keeps rough objects from gathering light from behind their tangent plane.
const toWorld = ( directionView, normalViewNode, roughnessNode ) => pow4( roughnessNode ).mix( directionView, normalViewNode ).normalize().transformDirection( cameraWorldMatrix );

const reflect = ( normalViewNode, roughnessNode ) => toWorld( positionViewDirection.negate().reflect( normalViewNode ), normalViewNode, roughnessNode );

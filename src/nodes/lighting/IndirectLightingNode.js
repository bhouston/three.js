import Node from '../core/Node.js';
import { context } from '../core/ContextNode.js';
import { Fn, If, nodeObject, float, vec3, vec4 } from '../tsl/TSLBase.js';
import { positionWorld } from '../accessors/Position.js';
import { normalWorld } from '../accessors/Normal.js';
import { cameraPosition } from '../accessors/Camera.js';
import { roughness } from '../core/PropertyNode.js';
import { materialEnvIntensity } from '../accessors/MaterialProperties.js';

/**
 * Resolves ordered, alternative estimates of indirect lighting. Providers receive
 * a query object and return a vec4: unweighted linear lighting in RGB and validity
 * coverage in A. Coverage is independent of brightness, intensity and occlusion:
 * a valid black sample must still consume coverage. Missing data returns vec4( 0 ).
 *
 * Each provider replaces its fraction of the remaining result. Providers must
 * estimate the same transport contribution; additional bounces must be composed
 * inside a provider, not added as fallback providers. Unresolved coverage stays black.
 *
 * @augments Node
 */
class IndirectLightingNode extends Node {

	constructor( providers ) {

		super( 'vec4' );

		if ( ! Array.isArray( providers ) || providers.some( provider => typeof provider !== 'function' ) ) {

			throw new TypeError( 'IndirectLightingNode: providers must be an array of functions returning lighting and coverage.' );

		}

		this.providers = Object.freeze( providers.slice() );

	}

	/**
	 * Evaluates this chain for a query. The result's RGB is coverage-weighted and
	 * its A is the resolved coverage. Use asProvider() when nesting chains.
	 *
	 * @param {Object} query - The world-space lighting query.
	 * @return {Node<vec4>} The resolved lighting and coverage.
	 */
	resolve( query ) {

		return Fn( () => {

			const value = vec3( 0 ).toVar();
			const remaining = float( 1 ).toVar();

			for ( const provider of this.providers ) {

				If( remaining.greaterThan( 0 ), () => {

					const sample = vec4( provider( query ) ).toVar();
					const weight = remaining.mul( sample.a.clamp() ).toConst();
					value.addAssign( sample.rgb.mul( weight ) );
					remaining.subAssign( weight );

				} );

			}

			return vec4( value, remaining.oneMinus() );

		} )();

	}

	/**
	 * Adapts this chain to a provider without applying coverage twice.
	 *
	 * @return {Function} A provider returning unweighted lighting and coverage.
	 */
	asProvider() {

		return query => Fn( () => {

			const result = this.sample( query ).toConst();
			return vec4( result.rgb.div( result.a.max( 1e-6 ) ), result.a );

		} )();

	}

	setup() {

		return this.sample();

	}

}

/**
 * An indirect diffuse chain. Queries contain position, normal and sampleIndex.
 * RGB is irradiance (not divided by pi), without material response or exposure.
 *
 * ```js
 * const diffuse = indirectDiffuse( [
 *   lightProbeGridDiffuse( probes ),
 *   environmentDiffuse( pmremTexture( environment ) )
 * ] );
 * renderer.contextNode = indirectLightingContext( diffuse );
 * const mapper = new LightMapper( renderer, 1024, diffuse );
 * ```
 *
 * Provider lists are immutable. Create a new chain when changing providers or
 * compiled source topology, such as the number of reflection probes.
 *
 * @augments IndirectLightingNode
 */
class IndirectDiffuseNode extends IndirectLightingNode {

	static get type() {

		return 'IndirectDiffuseNode';

	}

	constructor( providers = [] ) {

		super( providers );
		this.isIndirectDiffuseNode = true;

	}

	sample( { position = positionWorld, normal = normalWorld, sampleIndex = float( 0 ) } = {} ) {

		return this.resolve( { position, normal, sampleIndex } );

	}

}

/**
 * An indirect specular chain. Queries additionally contain viewDirection,
 * direction and roughness. RGB is filtered incoming radiance, before Fresnel.
 * Direction is the lobe sampling direction and can differ from reflect(-V, N),
 * e.g. for anisotropy, clearcoat or retroreflection.
 *
 * @augments IndirectLightingNode
 */
class IndirectSpecularNode extends IndirectLightingNode {

	static get type() {

		return 'IndirectSpecularNode';

	}

	constructor( providers = [] ) {

		super( providers );
		this.isIndirectSpecularNode = true;

	}

	sample( { position = positionWorld, normal = normalWorld, viewDirection = cameraPosition.sub( position ).normalize(), direction = viewDirection.negate().reflect( normal ), roughness: roughnessNode = roughness, sampleIndex = float( 0 ) } = {} ) {

		return this.resolve( { position, normal, viewDirection, direction, roughness: roughnessNode, sampleIndex } );

	}

}

export { IndirectDiffuseNode, IndirectSpecularNode };

export const indirectDiffuse = providers => nodeObject( new IndirectDiffuseNode( providers ) );
export const indirectSpecular = providers => nodeObject( new IndirectSpecularNode( providers ) );

/**
 * Installs independent indirect lighting chains on a renderer, material or pass
 * context. Null preserves the legacy path; an empty chain explicitly disables it.
 * Diffuse chains replace automatic GI hooks and diffuse probe grids.
 * Material light maps remain independent, additive irradiance contributions.
 * Direct lights and explicitly additive ambient/hemisphere lights are unchanged.
 *
 * @param {?IndirectDiffuseNode} [diffuse=null] - The diffuse chain.
 * @param {?IndirectSpecularNode} [specular=null] - The specular chain.
 * @return {ContextNode} The lighting context.
 */
export const indirectLightingContext = ( diffuse = null, specular = null ) => {

	if ( diffuse !== null && diffuse.isIndirectDiffuseNode !== true ) throw new TypeError( 'indirectLightingContext: expected an IndirectDiffuseNode.' );
	if ( specular !== null && specular.isIndirectSpecularNode !== true ) throw new TypeError( 'indirectLightingContext: expected an IndirectSpecularNode.' );
	return context( { indirectDiffuseNode: diffuse, indirectSpecularNode: specular } );

};

/**
 * Diffuse adapter for a filtered environment node, e.g. pmremTexture( texture ).
 *
 * @param {Node} environment - The filtered environment.
 * @param {Node<float>|number} [intensity=materialEnvIntensity] - Lighting intensity. Pass an explicit value when baking independently of a material or scene.
 * @return {Function} An irradiance provider.
 */
export const environmentDiffuse = ( environment, intensity = materialEnvIntensity ) => ( { normal } ) => vec4( environment.context( { getUV: () => normal, getTextureLevel: () => float( 1 ) } ).rgb.mul( Math.PI ).mul( intensity ), 1 );

/**
 * Diffuse adapter for sampled light-map irradiance, e.g. texture( atlas, uv( 1 ) ).
 * This is a surface/UV-space provider, not a world-space sampler for LightMapper.
 * Coverage is validity, independent of RGB and intensity; valid black blocks fallback.
 * Do not also assign the same irradiance to material.lightMap, which remains additive.
 *
 * @param {Node<vec3|vec4>} irradiance - Sampled linear irradiance, not divided by pi.
 * @param {Node<float>|number} [coverage=1] - Coverage of the sampled surface.
 * @param {Node<float>|number} [intensity=1] - Irradiance scale.
 * @return {Function} A surface-bound irradiance provider.
 */
export const lightMapDiffuse = ( irradiance, coverage = 1, intensity = 1 ) => () => vec4( irradiance.rgb.mul( intensity ), coverage );

/**
 * Specular adapter for a filtered environment node.
 *
 * @param {Node} environment - The filtered environment.
 * @param {Node<float>|number} [intensity=materialEnvIntensity] - Lighting intensity.
 * @return {Function} A radiance provider.
 */
export const environmentSpecular = ( environment, intensity = materialEnvIntensity ) => ( { direction, roughness: level } ) => vec4( environment.context( { getUV: () => direction, getTextureLevel: () => level } ).rgb.mul( intensity ), 1 );

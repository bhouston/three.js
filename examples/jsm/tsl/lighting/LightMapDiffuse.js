import { vec4, materialLightMap } from 'three/tsl';

/**
 * An indirect diffuse source for `builtinIndirectContext()` that returns the light map of a
 * material with a coverage of one, so it replaces the sources behind it. A material
 * without a light map is left to those sources.
 *
 * @tsl
 * @function
 * @returns {function(Object, NodeBuilder): ?Node<vec4>}
 */
export const lightMapDiffuse = () => ( query, { material } ) => {

	if ( material.lightMapNode ) return vec4( material.lightMapNode.rgb, 1 );

	return material.lightMap ? vec4( materialLightMap, 1 ) : null;

};

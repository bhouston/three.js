import { getShIrradianceAt, reference, uniformArray, vec4 } from 'three/tsl';

/**
 * An indirect diffuse source for a global light probe. Its coverage is one while
 * the probe is visible, including where its irradiance or intensity is zero.
 * Keep the probe out of the scene's light list to avoid adding its light twice.
 *
 * @tsl
 * @function
 * @param {LightProbe} probe - The light probe.
 * @returns {function(Object): Node<vec4>} The diffuse source.
 */
export const lightProbeDiffuse = ( probe ) => {

	const coefficients = uniformArray( probe.sh.coefficients );
	const intensity = reference( 'intensity', 'float', probe );
	const coverage = reference( 'visible', 'bool', probe ).toFloat();

	return ( { normal } ) => vec4( getShIrradianceAt( normal, coefficients ).mul( intensity, coverage ), coverage );

};

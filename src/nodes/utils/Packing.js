import { Fn, nodeObject, vec2, vec3, float } from '../tsl/TSLBase.js';
import { dot, sqrt, saturate } from '../math/MathNode.js';
import { warnOnce } from '../../utils.js';

/**
 * Encodes a nonzero normal direction as signed octahedral coordinates in [-1, 1].
 * Store the result directly in a two-channel floating-point texture (e.g. RG16F).
 * The input does not need to be normalized.
 *
 * @tsl
 * @function
 * @param {Node<vec3>} normal - The nonzero direction to encode.
 * @return {Node<vec2>} The signed octahedral coordinates.
 */
export const octEncode = /*@__PURE__*/ Fn( ( [ normal ] ) => {

	const p = normal.xy.div( normal.x.abs().add( normal.y.abs() ).add( normal.z.abs() ) ).toVar();
	// Zero must have a positive sign to preserve the negative Z pole and fold edges.
	const signNotZero = vec2( p.x.greaterThanEqual( 0 ).select( 1, - 1 ), p.y.greaterThanEqual( 0 ).select( 1, - 1 ) );
	return normal.z.lessThan( 0 ).select( p.yx.abs().oneMinus().mul( signNotZero ), p );

}, { normal: 'vec3', return: 'vec2' } );

/**
 * Decodes signed octahedral coordinates into a unit normal direction.
 *
 * @tsl
 * @function
 * @param {Node<vec2>} encoded - The signed octahedral coordinates in [-1, 1].
 * @return {Node<vec3>} The normalized direction.
 */
export const octDecode = /*@__PURE__*/ Fn( ( [ encoded ] ) => {

	const normal = vec3( encoded, float( 1 ).sub( encoded.x.abs() ).sub( encoded.y.abs() ) ).toVar();
	const t = normal.z.negate().max( 0 ).toVar();
	normal.x.addAssign( normal.x.greaterThanEqual( 0 ).select( t.negate(), t ) );
	normal.y.addAssign( normal.y.greaterThanEqual( 0 ).select( t.negate(), t ) );
	return normal.normalize();

}, { encoded: 'vec2', return: 'vec3' } );

/**
 * Packs a normal vector into a color value.
 *
 * @tsl
 * @function
 * @param {Node<vec3>} node - The direction to pack.
 * @return {Node<vec3>} The color.
 */
export const packNormalToRGB = ( node ) => nodeObject( node ).mul( 0.5 ).add( 0.5 );

/**
 * Unpacks a color value into a normal vector.
 *
 * @tsl
 * @function
 * @param {Node<vec3>} node - The color to unpack.
 * @return {Node<vec3>} The direction.
 */
export const unpackRGBToNormal = ( node ) => nodeObject( node ).mul( 2.0 ).sub( 1 );

/**
 * Unpacks a tangent space normal, reconstructing the Z component by projecting the X,Y coordinates onto the hemisphere.
 * The X,Y coordinates are expected to be in the [-1, 1] range.
 *
 * @tsl
 * @function
 * @param {Node<vec2>} xy - The X,Y coordinates of the normal.
 * @return {Node<vec3>} The resulting normal.
 */
export const unpackNormal = ( xy ) => vec3( xy, sqrt( saturate( float( 1.0 ).sub( dot( xy, xy ) ) ) ) );

/**
 * @tsl
 * @function
 * @deprecated since r185. Use {@link packNormalToRGB} instead.
 * @param {Node<vec3>} node - The direction to pack.
 * @returns {Node<vec3>}
 */
export const directionToColor = ( node ) => {

	warnOnce( 'TSL: "directionToColor()" has been renamed to "packNormalToRGB()".' ); // @deprecated r185

	return packNormalToRGB( node );

};

/**
 * @tsl
 * @function
 * @deprecated since r185. Use {@link unpackRGBToNormal} instead.
 * @param {Node<vec3>} node - The color to unpack.
 * @returns {Node<vec3>}
 */
export const colorToDirection = ( node ) => {

	warnOnce( 'TSL: "colorToDirection()" has been renamed to "unpackRGBToNormal()".' ); // @deprecated r185

	return unpackRGBToNormal( node );

};

import { Fn, float, vec2, vec3 } from 'three/tsl';

/**
 * @module OctahedralNormal
 * @three_import import { packNormalToOctahedral, unpackOctahedralToNormal } from 'three/addons/tsl/utils/OctahedralNormal.js';
 */

/**
 * Encodes a nonzero direction as signed octahedral coordinates in `[-1, 1]`
 * (Cigolle et al., {@link https://jcgt.org/published/0003/02/01/ A Survey of Efficient Representations for Independent Unit Vectors}, 2014).
 * Store the result in a two-channel texture such as RG16F, or RG16Snorm for higher precision. The input does not need to be normalized.
 *
 * @tsl
 * @function
 * @param {Node<vec3>} normal - The nonzero direction to encode.
 * @return {Node<vec2>} The signed octahedral coordinates.
 */
export const packNormalToOctahedral = /*@__PURE__*/ Fn( ( [ normal ] ) => {

	const p = normal.xy.div( normal.x.abs().add( normal.y.abs() ).add( normal.z.abs() ) ).toVar();
	// Zero must have a positive sign to preserve the negative Z pole and fold edges.
	const signNotZero = vec2( p.x.greaterThanEqual( 0 ).select( 1, - 1 ), p.y.greaterThanEqual( 0 ).select( 1, - 1 ) );
	return normal.z.lessThan( 0 ).select( p.yx.abs().oneMinus().mul( signNotZero ), p );

}, { normal: 'vec3', return: 'vec2' } );

/**
 * Decodes signed octahedral coordinates into a unit direction.
 *
 * @tsl
 * @function
 * @param {Node<vec2>} encoded - The signed octahedral coordinates in `[-1, 1]`.
 * @return {Node<vec3>} The normalized direction.
 */
export const unpackOctahedralToNormal = /*@__PURE__*/ Fn( ( [ encoded ] ) => {

	// using Rune Stubbe's optimized decoder: https://x.com/Stubbesaurus/status/937994790553227264
	const normal = vec3( encoded, float( 1 ).sub( encoded.x.abs() ).sub( encoded.y.abs() ) ).toVar();
	const t = normal.z.negate().max( 0 ).toVar();
	normal.x.addAssign( normal.x.greaterThanEqual( 0 ).select( t.negate(), t ) );
	normal.y.addAssign( normal.y.greaterThanEqual( 0 ).select( t.negate(), t ) );
	return normal.normalize();

}, { encoded: 'vec2', return: 'vec3' } );

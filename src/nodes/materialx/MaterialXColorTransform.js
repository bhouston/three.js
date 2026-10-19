import { bvec3, vec3, Fn } from '../tsl/TSLBase.js';
import { greaterThan } from '../math/OperatorNode.js';
import { max, pow, mix } from '../math/MathNode.js';

export const mx_srgb_texture_to_lin_rec709 = /*@__PURE__*/ Fn( ( [ color_immutable ] ) => {

	const color = vec3( color_immutable ).toVar();
	const isAbove = bvec3( greaterThan( color.abs(), vec3( 0.04045 ) ) ).toVar();
	const linSeg = vec3( color.div( 12.92 ) ).toVar();
	const powSeg = vec3( pow( max( color.abs().add( vec3( 0.055 ) ), vec3( 0.0 ) ).div( 1.055 ), vec3( 2.4 ) ).mul( color.sign() ) ).toVar();

	return mix( linSeg, powSeg, isAbove );

}, { color: 'vec3', return: 'vec3' } );

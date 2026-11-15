import { Fn, float, vec3, vec4, ivec2, int, uv, floor, abs, max, min, exp2, nodeObject } from 'three/tsl';

/**
 * Inline RCAS, matching SharpenNode while preserving dynamic texture references.
 * Can be fused into the final display shader, avoiding a separate full-size pass.
 * Reference: https://gpuopen.com/fidelityfx-superresolution/
 *
 * @param {TextureNode} inputTextureNode - Input texture, including ping-pong pass textures.
 * @param {number|Node<float>} [sharpness=0.2] - RCAS sharpening parameter.
 * @param {boolean|Node<bool>} [denoise=false] - Attenuate sharpening in noisy areas.
 * @returns {Node<vec4>} Sharpened color, with the source alpha.
 */
export const researchSharpen = ( inputTextureNode, sharpness = 0.2, denoise = false ) => {

	const sharpnessNode = nodeObject( sharpness );
	const denoiseNode = nodeObject( denoise );

	return Fn( () => {

		const targetUV = uv();
		const texSize = inputTextureNode.size();

		const p = ivec2( int( floor( targetUV.x.mul( texSize.x ) ) ), int( floor( targetUV.y.mul( texSize.y ) ) ) ).toConst();

		const e = inputTextureNode.load( p );
		const b = inputTextureNode.load( p.add( ivec2( 0, - 1 ) ) );
		const d = inputTextureNode.load( p.add( ivec2( - 1, 0 ) ) );
		const f = inputTextureNode.load( p.add( ivec2( 1, 0 ) ) );
		const h = inputTextureNode.load( p.add( ivec2( 0, 1 ) ) );

		// Approximate luminance (luma times 2).

		const luma = ( s ) => s.g.add( s.b.add( s.r ).mul( 0.5 ) );

		const bL = luma( b );
		const dL = luma( d );
		const eL = luma( e );
		const fL = luma( f );
		const hL = luma( h );

		// Sharpening amount from user parameter.

		const con = exp2( sharpnessNode.negate() ).toConst();

		// Min and max of ring.

		const mn4 = min( min( b.rgb, d.rgb ), min( f.rgb, h.rgb ) ).toConst();
		const mx4 = max( max( b.rgb, d.rgb ), max( f.rgb, h.rgb ) ).toConst();

		// Compute adaptive lobe weight.
		// Limiters based on how much sharpening the local contrast can tolerate.

		const RCAS_LIMIT = float( 0.25 - 1.0 / 16.0 ).toConst();

		const hitMin = min( mn4, e.rgb ).div( mx4.mul( 4.0 ) ).toConst();
		const hitMax = vec3( 1.0 ).sub( max( mx4, e.rgb ) ).div( mn4.mul( 4.0 ).sub( 4.0 ) ).toConst();
		const lobeRGB = max( hitMin.negate(), hitMax ).toConst();

		const lobe = max(
			RCAS_LIMIT.negate(),
			min( max( lobeRGB.r, max( lobeRGB.g, lobeRGB.b ) ), float( 0.0 ) )
		).mul( con ).toConst();

		// Noise attenuation.

		const nz = bL.add( dL ).add( fL ).add( hL ).mul( 0.25 ).sub( eL ).toConst();
		const nzRange = max( max( bL, dL ), max( eL, max( fL, hL ) ) ).sub( min( min( bL, dL ), min( eL, min( fL, hL ) ) ) ).toConst();
		const nzFactor = float( 1.0 ).sub( abs( nz ).div( max( nzRange, float( 1.0 / 65536.0 ) ) ).saturate().mul( 0.5 ) ).toConst();

		const effectiveLobe = denoiseNode.equal( true ).select( lobe.mul( nzFactor ), lobe ).toConst();

		// Resolve: weighted blend of cross neighbors and center.

		const result = b.rgb.add( d.rgb ).add( f.rgb ).add( h.rgb ).mul( effectiveLobe ).add( e.rgb )
			.div( effectiveLobe.mul( 4.0 ).add( 1.0 ) ).toConst();

		return vec4( result, e.a );

	} )();

};

import { context, exp2, uv } from 'three/tsl';

/**
 * Supplies material UV gradients corresponding to a finer texture footprint.
 * Assign the result to the low resolution scene pass's `contextNode`, keeping
 * the native reference and guide passes unchanged. A bias of -log2(factor)
 * uses the display resolution footprint; a further -1 is often used by FSR2.
 * The bias may be a uniform so resolution changes do not rebuild materials.
 *
 * This affects implicit material texture samples. Explicit texture levels and
 * environment-map roughness LODs retain their own sampling rules.
 *
 * @param {Node<float>} biasNode - Texture mip bias in mip levels.
 * @returns {ContextNode} The context for a PassNode.
 */
export function upscalerMipBias( biasNode ) {

	return context( {
		getTextureGradient: ( textureNode ) => {

			const texture = textureNode.value;
			if ( texture.isRenderTargetTexture || texture.isDepthTexture || texture.isCubeTexture ) return null;

			const coordinate = uv( texture.channel );
			const scale = exp2( biasNode );
			return [ coordinate.dFdx().mul( scale ), coordinate.dFdy().mul( scale ) ];

		}
	} );

}

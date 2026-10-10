import { DataUtils, FloatType, HalfFloatType, Texture } from 'three';

/**
 * A glTF exporter plugin for the vendor extension `MOZ_lightmap`.
 * Light maps must contain finite, nonnegative linear irradiance. Data textures
 * must use RGBAFormat with unsigned byte, float or half-float pixels.
 * Float and half-float textures are scaled into an 8-bit linear PNG,
 * with the scale stored in the intensity.
 * This preserves HDR range with 8-bit precision. Render target textures must
 * be read back into data textures before export. Compressed textures use
 * GLTFExporter.setTextureUtils().
 *
 * ```js
 * exporter.register( writer => new GLTFLightMapExporterExtension( writer ) );
 * ```
 *
 * @three_import import { GLTFLightMapExporterExtension } from 'three/addons/exporters/GLTFLightMapExporterExtension.js';
 */
class GLTFLightMapExporterExtension {

	/**
	 * Constructs a light map exporter plugin.
	 *
	 * @param {GLTFWriter} writer - The glTF writer.
	 */
	constructor( writer ) {

		this.writer = writer;
		this.name = 'MOZ_lightmap';
		this._textures = new Map();

	}

	/**
	 * Writes a material's light map.
	 *
	 * @param {Material} material - The material.
	 * @param {Object} materialDef - The glTF material definition.
	 * @return {Promise<void>} Resolves after the light map has been written.
	 */
	async writeMaterialAsync( material, materialDef ) {

		const map = material.lightMap;
		if ( ! map ) return;

		if ( ! this._textures.has( map ) ) this._textures.set( map, this._writeTexture( map ) );
		const { index, scale } = await this._textures.get( map );

		const extension = {
			index,
			texCoord: map.channel,
			intensity: material.lightMapIntensity * scale
		};

		this.writer.applyTextureTransform( extension, map );
		materialDef.extensions = materialDef.extensions || {};
		materialDef.extensions[ this.name ] = extension;
		this.writer.extensionsUsed[ this.name ] = true;

	}

	async _writeTexture( map ) {

		if ( map.isCompressedTexture ) return { index: await this.writer.processTextureAsync( map ), scale: 1 };

		let image = map.image;

		let scale = 1;
		let texture = map;

		if ( image.data !== undefined && ( map.type === FloatType || map.type === HalfFloatType ) ) {

			const decode = map.type === HalfFloatType ? DataUtils.fromHalfFloat : value => value;
			const data = image.data;

			for ( let i = 0; i < data.length; i ++ ) {

				if ( i % 4 === 3 ) continue;
				const value = decode( data[ i ] );
				scale = Math.max( scale, value );

			}

			const bytes = new Uint8Array( data.length );
			for ( let i = 0; i < bytes.length; i ++ ) bytes[ i ] = i % 4 === 3 ? 255 : Math.round( decode( data[ i ] ) / scale * 255 );
			image = { data: bytes, width: image.width, height: image.height };

		}

		if ( image.data !== undefined ) {

			// Export through a canvas so flipY also applies to data textures.
			const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( image.width, image.height ) : document.createElement( 'canvas' );
			canvas.width = image.width;
			canvas.height = image.height;
			canvas.getContext( '2d' ).putImageData( new ImageData( new Uint8ClampedArray( image.data ), image.width, image.height ), 0, 0 );
			texture = new Texture( canvas );
			for ( const property of [ 'name', 'flipY', 'minFilter', 'magFilter', 'wrapS', 'wrapT' ] ) texture[ property ] = map[ property ];

		}

		return { index: await this.writer.processTextureAsync( texture ), scale };

	}

}

export { GLTFLightMapExporterExtension };

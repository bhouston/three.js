import {
	DataTexture,
	DataUtils,
	FileLoader,
	HalfFloatType,
	LinearFilter,
	LinearSRGBColorSpace,
	Loader,
	RGBAFormat
} from 'three';

/**
 * Experimental native AVIF/JPEG XL HDR decoder. Requires float16 Canvas2D and
 * globalHDRHeadroom. Unsupported browsers fail rather than silently return SDR.
 * Output is linear sRGB, relative to the browser's reference white, not physical
 * radiance. This prototype does not define a glTF extension or parse HDR metadata.
 *
 * @augments Loader
 * @three_import import { NativeHDRLoader } from 'three/addons/loaders/NativeHDRLoader.js';
 */
class NativeHDRLoader extends Loader {

	/**
	 * Loads a native HDR image into a half-float data texture.
	 * @param {string} url - Image URL.
	 * @param {function(DataTexture)} onLoad - Completion callback.
	 * @param {Function} onProgress - Download progress callback.
	 * @param {Function} onError - Error callback.
	 * @return {DataTexture} Placeholder texture, populated after decoding.
	 */
	load( url, onLoad, onProgress, onError ) {

		const texture = new DataTexture();
		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		loader.setResponseType( 'arraybuffer' );

		// Keep LoadingManager pending during asynchronous browser decoding as well.
		const decodeURL = this.manager.resolveURL( this.path + url ) + '#native-hdr-decode';
		this.manager.itemStart( decodeURL );
		const fail = ( error ) => {

			try {

				if ( onError ) onError( error );
				else console.error( error );
				this.manager.itemError( decodeURL );

			} finally {

				this.manager.itemEnd( decodeURL );

			}

		};

		loader.load( url, ( buffer ) => {

			this.parse( buffer ).then( ( decoded ) => {

				texture.copy( decoded );
				texture.needsUpdate = true;
				try {

					if ( onLoad ) onLoad( texture );

				} finally {

					this.manager.itemEnd( decodeURL );

				}

			}, fail );

		}, onProgress, fail );
		return texture;

	}

	/**
	 * Decodes image bytes using the browser, preserving maximum HDR headroom.
	 * @param {ArrayBuffer} buffer - AVIF or JPEG XL bytes.
	 * @return {Promise<DataTexture>} Linear sRGB half-float texture.
	 */
	async parse( buffer ) {

		const canvas = document.createElement( 'canvas' );
		const context = canvas.getContext( '2d', { colorSpace: 'srgb', colorType: 'float16' } );

		if ( ! context || context.getContextAttributes().colorType !== 'float16' ||
			! ( 'globalHDRHeadroom' in context ) ) {

			throw new Error( 'NativeHDRLoader: HDR-preserving native decoding requires float16 Canvas2D and experimental globalHDRHeadroom support.' );

		}

		const url = URL.createObjectURL( new Blob( [ buffer ] ) );
		const image = new Image();

		try {

			image.src = url;
			await image.decode();
			canvas.width = image.naturalWidth;
			canvas.height = image.naturalHeight;
			context.globalHDRHeadroom = Infinity;
			context.drawImage( image, 0, 0 );

			const pixels = context.getImageData( 0, 0, canvas.width, canvas.height, {
				colorSpace: 'srgb', pixelFormat: 'rgba-float16'
			} );
			if ( pixels.pixelFormat !== 'rgba-float16' ) {

				throw new Error( 'NativeHDRLoader: float16 readback unavailable.' );

			}

			const data = new Uint16Array( pixels.data.length );
			let maxLinear = 0;

			for ( let i = 0; i < data.length; i ++ ) {

				let value = pixels.data[ i ];
				if ( ! Number.isFinite( value ) ) {

					throw new Error( 'NativeHDRLoader: browser returned non-finite HDR pixels.' );

				}

				if ( i % 4 !== 3 ) {

					// Extended sRGB inverse transfer, including negative wide-gamut values.
					const magnitude = Math.abs( value );
					value = Math.sign( value ) * ( magnitude <= 0.04045 ? magnitude / 12.92 : Math.pow( ( magnitude + 0.055 ) / 1.055, 2.4 ) );
					maxLinear = Math.max( maxLinear, value );

				}

				if ( Math.abs( value ) > 65504 ) {

					throw new Error( 'NativeHDRLoader: pixel exceeds half-float range.' );

				}

				data[ i ] = DataUtils.toHalfFloat( value );

			}

			const texture = new DataTexture( data, canvas.width, canvas.height, RGBAFormat, HalfFloatType );
			texture.colorSpace = LinearSRGBColorSpace;
			texture.minFilter = LinearFilter;
			texture.magFilter = LinearFilter;
			texture.flipY = true;
			texture.userData.nativeHDR = { maxLinear, headroom: 'maximum', referenceWhite: 'browser-defined' };
			texture.needsUpdate = true;
			return texture;

		} finally {

			URL.revokeObjectURL( url );
			canvas.width = canvas.height = 0;

		}

	}

}

export { NativeHDRLoader };

import {
	ClampToEdgeWrapping,
	DataUtils,
	DataTexture,
	FileLoader,
	HalfFloatType,
	LinearFilter,
	LinearMipmapLinearFilter,
	Loader,
	NoColorSpace,
	RGBAFormat,
	RepeatWrapping
} from 'three';

const FORMAT = 'three-neural-appearance';
const VERSION = 1;
const LATENT_CHANNELS = 8;
const LATENT_TEXTURES = 2;
const CHANNELS_PER_TEXTURE = 4;

/**
 * A loader for compact neural appearance material assets.
 *
 * The format stores an 8D latent texture as two RGBA16F textures and a small
 * MLP decoder that evaluates the learned BRDF for a pair of tangent-space
 * directions. Training is intentionally offline; this loader only prepares
 * runtime data for {@link NeuralAppearanceNodeMaterial}.
 *
 * @augments Loader
 * @three_import import { NeuralAppearanceLoader } from 'three/addons/loaders/NeuralAppearanceLoader.js';
 */
class NeuralAppearanceLoader extends Loader {

	/**
	 * Constructs a new neural appearance loader.
	 *
	 * @param {LoadingManager} [manager] - The loading manager.
	 */
	constructor( manager ) {

		super( manager );

	}

	/**
	 * Starts loading from the given URL and passes the parsed neural material
	 * data to the `onLoad()` callback.
	 *
	 * @param {string} url - The path/URL of the JSON file to load.
	 * @param {function(Object)} onLoad - Executed when loading has finished.
	 * @param {onProgressCallback} onProgress - Executed while loading progresses.
	 * @param {onErrorCallback} onError - Executed when errors occur.
	 */
	load( url, onLoad, onProgress, onError ) {

		const scope = this;

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setResponseType( 'json' );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		loader.load( url, function ( json ) {

			try {

				onLoad( scope.parse( json ) );

			} catch ( e ) {

				if ( onError ) {

					onError( e );

				} else {

					console.error( e );

				}

				scope.manager.itemError( url );

			}

		}, onProgress, onError );

	}

	/**
	 * Parses a neural appearance material manifest.
	 *
	 * @param {(Object|string)} data - The JSON manifest, either parsed or as a string.
	 * @return {Object} Parsed material data for `NeuralAppearanceNodeMaterial`.
	 */
	parse( data ) {

		const manifest = ( typeof data === 'string' ) ? JSON.parse( data ) : data;

		validateManifest( manifest );

		const latentTextures = manifest.latents.textures.map( ( textureConfig, index ) => {

			const mipmaps = textureConfig.mipmaps.map( ( mipmap, mipLevel ) => createMipmap( mipmap, `latents.textures[${ index }].mipmaps[${ mipLevel }]` ) );
			const firstMipmap = mipmaps[ 0 ];
			const texture = new DataTexture(
				firstMipmap.data,
				firstMipmap.width,
				firstMipmap.height,
				RGBAFormat,
				HalfFloatType,
				undefined,
				textureConfig.wrap === 'repeat' ? RepeatWrapping : ClampToEdgeWrapping,
				textureConfig.wrap === 'repeat' ? RepeatWrapping : ClampToEdgeWrapping,
				LinearFilter,
				mipmaps.length > 1 ? LinearMipmapLinearFilter : LinearFilter,
				undefined,
				NoColorSpace
			);

			texture.generateMipmaps = false;
			texture.mipmaps = mipmaps;
			texture.needsUpdate = true;

			return texture;

		} );

		return {
			isNeuralAppearanceData: true,
			name: manifest.name || '',
			latentTextures,
			latentWidth: latentTextures[ 0 ].image.width,
			latentHeight: latentTextures[ 0 ].image.height,
			mipLevels: latentTextures[ 0 ].mipmaps.length,
			wrap: manifest.latents.wrap || 'repeat',
			decoder: normalizeDecoder( manifest.decoder ),
			referenceEvaluations: manifest.referenceEvaluations || []
		};

	}

}

function createMipmap( mipmap, path ) {

	assertInteger( mipmap.width, `${ path }.width`, 1 );
	assertInteger( mipmap.height, `${ path }.height`, 1 );

	const expectedLength = mipmap.width * mipmap.height * CHANNELS_PER_TEXTURE;
	const data = toFloat32Array( mipmap.data, `${ path }.data` );

	if ( data.length !== expectedLength ) {

		throw new Error( `THREE.NeuralAppearanceLoader: ${ path }.data length must be ${ expectedLength }.` );

	}

	return {
		data: toHalfFloatArray( data ),
		width: mipmap.width,
		height: mipmap.height
	};

}

function toHalfFloatArray( data ) {

	const result = new Uint16Array( data.length );

	for ( let i = 0; i < data.length; i ++ ) {

		result[ i ] = DataUtils.toHalfFloat( data[ i ] );

	}

	return result;

}

function normalizeDecoder( decoder ) {

	const rotation = decoder.rotation || null;

	if ( rotation !== null ) {

		assertInteger( rotation.inputSize, 'decoder.rotation.inputSize', LATENT_CHANNELS, LATENT_CHANNELS );
		assertInteger( rotation.outputSize, 'decoder.rotation.outputSize', 12, 12 );
		rotation.weights = validateArray( rotation.weights, 'decoder.rotation.weights', rotation.inputSize * rotation.outputSize );

	}

	const layers = decoder.layers.map( ( layer, index ) => {

		const path = `decoder.layers[${ index }]`;
		assertInteger( layer.inputSize, `${ path }.inputSize`, 1 );
		assertInteger( layer.outputSize, `${ path }.outputSize`, 1 );

		const activation = layer.activation || 'linear';
		if ( activation !== 'linear' && activation !== 'relu' ) {

			throw new Error( `THREE.NeuralAppearanceLoader: Unsupported ${ path }.activation "${ activation }".` );

		}

		return {
			inputSize: layer.inputSize,
			outputSize: layer.outputSize,
			activation,
			weights: validateArray( layer.weights, `${ path }.weights`, layer.inputSize * layer.outputSize ),
			biases: validateArray( layer.biases || [], `${ path }.biases`, layer.outputSize )
		};

	} );

	for ( let i = 1; i < layers.length; i ++ ) {

		if ( layers[ i ].inputSize !== layers[ i - 1 ].outputSize ) {

			throw new Error( `THREE.NeuralAppearanceLoader: decoder.layers[${ i }].inputSize does not match the previous output size.` );

		}

	}

	const outputActivation = decoder.outputActivation || { type: 'linear' };

	if ( outputActivation.type !== 'linear' && outputActivation.type !== 'exp' && outputActivation.type !== 'scaledSigmoid' ) {

		throw new Error( `THREE.NeuralAppearanceLoader: Unsupported decoder.outputActivation.type "${ outputActivation.type }".` );

	}

	return {
		inputSize: decoder.inputSize,
		rotation,
		layers,
		outputActivation
	};

}

function validateManifest( manifest ) {

	if ( manifest === null || typeof manifest !== 'object' ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: Manifest must be an object.' );

	}

	if ( manifest.format !== FORMAT ) {

		throw new Error( `THREE.NeuralAppearanceLoader: Unsupported format "${ manifest.format }".` );

	}

	if ( manifest.version !== VERSION ) {

		throw new Error( `THREE.NeuralAppearanceLoader: Unsupported version ${ manifest.version }.` );

	}

	if ( ! manifest.latents || ! Array.isArray( manifest.latents.textures ) ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: Manifest must define latents.textures.' );

	}

	if ( manifest.latents.channels !== LATENT_CHANNELS ) {

		throw new Error( `THREE.NeuralAppearanceLoader: latents.channels must be ${ LATENT_CHANNELS }.` );

	}

	if ( manifest.latents.textures.length !== LATENT_TEXTURES ) {

		throw new Error( `THREE.NeuralAppearanceLoader: latents.textures must contain ${ LATENT_TEXTURES } RGBA texture definitions.` );

	}

	const mipCount = manifest.latents.textures[ 0 ].mipmaps.length;

	for ( let textureIndex = 0; textureIndex < manifest.latents.textures.length; textureIndex ++ ) {

		const texture = manifest.latents.textures[ textureIndex ];

		if ( texture.wrap !== undefined && texture.wrap !== 'repeat' && texture.wrap !== 'clamp' ) {

			throw new Error( `THREE.NeuralAppearanceLoader: Unsupported latents.textures[${ textureIndex }].wrap "${ texture.wrap }".` );

		}

		if ( ! Array.isArray( texture.mipmaps ) || texture.mipmaps.length === 0 ) {

			throw new Error( `THREE.NeuralAppearanceLoader: latents.textures[${ textureIndex }].mipmaps must be a non-empty array.` );

		}

		if ( texture.mipmaps.length !== mipCount ) {

			throw new Error( 'THREE.NeuralAppearanceLoader: Latent textures must have the same number of mip levels.' );

		}

	}

	if ( ! manifest.decoder || ! Array.isArray( manifest.decoder.layers ) || manifest.decoder.layers.length === 0 ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: Manifest must define decoder.layers.' );

	}

	assertInteger( manifest.decoder.inputSize, 'decoder.inputSize', 1 );

	if ( manifest.decoder.inputSize !== 20 ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: Only 20-input decoders with two learned shading frames are supported.' );

	}

}

function validateArray( array, path, expectedLength ) {

	const result = toFloat32Array( array, path );

	if ( result.length !== expectedLength ) {

		throw new Error( `THREE.NeuralAppearanceLoader: ${ path } length must be ${ expectedLength }.` );

	}

	return result;

}

function toFloat32Array( array, path ) {

	if ( array instanceof Float32Array ) return array;
	if ( typeof Float16Array !== 'undefined' && array instanceof Float16Array ) return new Float32Array( array );
	if ( Array.isArray( array ) ) return Float32Array.from( array );

	throw new Error( `THREE.NeuralAppearanceLoader: ${ path } must be an array.` );

}

function assertInteger( value, path, min, max = Infinity ) {

	if ( Number.isInteger( value ) === false || value < min || value > max ) {

		throw new Error( `THREE.NeuralAppearanceLoader: ${ path } must be an integer in [${ min }, ${ max }].` );

	}

}

export { NeuralAppearanceLoader };

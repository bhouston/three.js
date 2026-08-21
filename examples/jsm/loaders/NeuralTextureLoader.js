import { FileLoader, Loader } from 'three';
import { FORMAT, VERSION } from '../neural-texture/NeuralTextureManifest.js';
import { decodeUint8Base64, decodeMLPLayersBase64 } from '../neural/NeuralBinaryCodec.js';

/**
 * A loader for compact `.neuralTexture` neural texture assets (see
 * NeuralTextureManifest.js) - a quantized multiresolution latent grid
 * pyramid plus a float16-packed MLP decoder.
 *
 * `parse()`/`parseManifestObject()` reconstructs the exact CPU model object
 * shape `NeuralTextureNodeMaterial.js`'s `buildLevelTextures`/
 * `evaluateNeuralTextureRaw` already expect from a live `NeuralTextureTrainer`
 * run (`{ channels, levels, grids, decoder, outputChannels, peOctaves,
 * inputEncoding }`), so that module needs no changes to consume a loaded
 * model.
 *
 * @augments Loader
 * @three_import import { NeuralTextureLoader } from 'three/addons/loaders/NeuralTextureLoader.js';
 */
class NeuralTextureLoader extends Loader {

	/**
	 * Constructs a new neural texture loader.
	 *
	 * @param {LoadingManager} [manager] - The loading manager.
	 */
	constructor( manager ) {

		super( manager );

	}

	/**
	 * Starts loading from the given URL and passes the parsed neural texture
	 * CPU model to the `onLoad()` callback.
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
	 * Parses a `.neuralTexture` manifest.
	 *
	 * @param {(Object|string)} data - The JSON manifest, either parsed or as a string.
	 * @return {Object} The decoded CPU model, ready for `NeuralTextureNodeMaterial`.
	 */
	parse( data ) {

		const manifest = ( typeof data === 'string' ) ? JSON.parse( data ) : data;

		return this.parseManifestObject( manifest );

	}

	/**
	 * Decodes an already-parsed `.neuralTexture` manifest object into a CPU
	 * model. Factored out from `parse()` (rather than only supporting a
	 * whole-file string/object) so `NeuralMaterialLoader.js` can decode a
	 * `.neuralMaterial` manifest's embedded `texture` block through the exact
	 * same path, without round-tripping it through JSON first.
	 *
	 * @param {Object} manifest - A parsed `.neuralTexture` manifest object.
	 * @return {Object} The decoded CPU model.
	 */
	parseManifestObject( manifest ) {

		validateManifest( manifest );

		const grids = manifest.latents.levels.map( ( level, index ) => decodeLevel( level, `latents.levels[${ index }]` ) );
		const decoderLayers = decodeMLPLayersBase64( manifest.mlp );

		const peOctaves = Number.isInteger( manifest.peOctaves ) ? manifest.peOctaves : 0;
		const inputEncoding = manifest.inputEncoding !== undefined ?
			manifest.inputEncoding : ( peOctaves > 0 ? 'positional' : 'none' );

		return {
			channels: manifest.latents.channelsPerLevel,
			levels: grids.length,
			grids,
			decoder: { layers: decoderLayers },
			outputChannels: manifest.outputChannels !== undefined ?
				manifest.outputChannels : decoderLayers[ decoderLayers.length - 1 ].outputSize,
			peOctaves,
			inputEncoding,
			wrap: manifest.latents.wrap || 'repeat',
			name: manifest.name || ''
		};

	}

}

function decodeLevel( level, path ) {

	assertInteger( level.width, `${ path }.width`, 1 );
	assertInteger( level.height, `${ path }.height`, 1 );
	assertInteger( level.channels, `${ path }.channels`, 1, 4 );

	if ( level.dtype !== 'uint8' ) {

		throw new Error( `THREE.NeuralTextureLoader: Unsupported ${ path }.dtype "${ level.dtype }".` );

	}

	const expectedLength = level.width * level.height * level.channels;
	const data = decodeUint8Base64( level.dataBase64, level.min, level.max, expectedLength );

	return { width: level.width, height: level.height, channels: level.channels, data };

}

function validateManifest( manifest ) {

	if ( manifest === null || typeof manifest !== 'object' ) {

		throw new Error( 'THREE.NeuralTextureLoader: Manifest must be an object.' );

	}

	if ( manifest.format !== FORMAT ) {

		throw new Error( `THREE.NeuralTextureLoader: Unsupported format "${ manifest.format }" (expected "${ FORMAT }").` );

	}

	if ( manifest.version !== VERSION ) {

		throw new Error( `THREE.NeuralTextureLoader: Unsupported version ${ manifest.version } (expected ${ VERSION }).` );

	}

	if ( ! manifest.latents || ! Array.isArray( manifest.latents.levels ) || manifest.latents.levels.length === 0 ) {

		throw new Error( 'THREE.NeuralTextureLoader: Manifest must define a non-empty latents.levels array.' );

	}

	if ( ! manifest.mlp || typeof manifest.mlp.dataBase64 !== 'string' || ! Array.isArray( manifest.mlp.layout ) ) {

		throw new Error( 'THREE.NeuralTextureLoader: Manifest must define mlp.layout and mlp.dataBase64.' );

	}

}

function assertInteger( value, path, min, max = Infinity ) {

	if ( Number.isInteger( value ) === false || value < min || value > max ) {

		throw new Error( `THREE.NeuralTextureLoader: ${ path } must be an integer in [${ min }, ${ max }].` );

	}

}

export { NeuralTextureLoader };

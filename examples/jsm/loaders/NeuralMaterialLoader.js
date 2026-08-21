import { FileLoader, Loader } from 'three';
import { FORMAT, VERSION } from '../neural-material/NeuralMaterialManifest.js';
import { getChannel, layoutChannels, decodeConstantValues } from '../neural-material/NeuralMaterialFormat.js';
import { NeuralTextureLoader } from './NeuralTextureLoader.js';

/**
 * A loader for compact `.neuralMaterial` neural material assets (see
 * NeuralMaterialManifest.js) - one embedded `.neuralTexture`-shaped
 * quantized grid+MLP model (decoded via `NeuralTextureLoader.
 * parseManifestObject`), plus the active/constant PBR channel layout needed
 * to slice its packed output back into named channels.
 *
 * `parse()` reconstructs `{ cpuModel, channelClassification }` - exactly the
 * two arguments `new NeuralMaterialNodeMaterial( cpuModel,
 * channelClassification, options )` expects, and the same shape
 * `NeuralMaterialSource.classifyMaterialChannels` produces from a live
 * material - so `NeuralMaterialNodeMaterial.js` needs no changes to consume
 * a loaded material.
 *
 * @augments Loader
 * @three_import import { NeuralMaterialLoader } from 'three/addons/loaders/NeuralMaterialLoader.js';
 */
class NeuralMaterialLoader extends Loader {

	/**
	 * Constructs a new neural material loader.
	 *
	 * @param {LoadingManager} [manager] - The loading manager.
	 */
	constructor( manager ) {

		super( manager );

		this._textureLoader = new NeuralTextureLoader( manager );

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
	 * Parses a `.neuralMaterial` manifest.
	 *
	 * @param {(Object|string)} data - The JSON manifest, either parsed or as a string.
	 * @return {Object} `{ name, cpuModel, channelClassification }`, ready for `NeuralMaterialNodeMaterial`.
	 */
	parse( data ) {

		const manifest = ( typeof data === 'string' ) ? JSON.parse( data ) : data;

		validateManifest( manifest );

		const cpuModel = this._textureLoader.parseManifestObject( manifest.texture );
		const channelClassification = decodeChannelClassification( manifest.channels, manifest.renderFlags );

		return {
			name: manifest.name || '',
			cpuModel,
			channelClassification
		};

	}

}

function decodeChannelClassification( channels, renderFlags ) {

	const activeList = channels.activeKeys.map( ( key ) => getChannel( key ) );
	const { channels: activeChannels, totalChannels, packCount } = layoutChannels( activeList );

	// Deliberately *not* run through `decodeConstantValues` here -
	// `constantValues` is what feeds each channel's `applyConstant` (see
	// NeuralMaterialNodeMaterial.js), which - for `attenuationDistance` -
	// expects the same finite, JSON/shader-safe sentinel `resolveConstant`
	// encoded on export (see NeuralMaterialFormat.js), not the real
	// `Infinity` it stands in for; `float(Infinity)` is not valid generated
	// shader source. This also keeps a loaded material's `constantValues`
	// shaped exactly like a freshly-classified live material's (`
	// classifyMaterialChannels` never decodes it either) - both are always
	// the encoded form. A caller that wants the true semantic values back
	// (e.g. for display) should call `decodeConstantValues` itself on a
	// separate copy, not on what gets handed to `NeuralMaterialNodeMaterial`.
	//
	// `renderFlags` is `null`/`undefined` for a manifest saved before this
	// field existed (or built without a source material) - passed through
	// as-is, since `NeuralMaterialNodeMaterial`'s constructor already treats
	// a falsy `renderFlags` as "keep the class default" (see
	// NeuralMaterialSource.resolveRenderFlags's doc comment).
	return { activeChannels, totalChannels, packCount, constantValues: channels.constantValues || {}, renderFlags: renderFlags || null };

}

function validateManifest( manifest ) {

	if ( manifest === null || typeof manifest !== 'object' ) {

		throw new Error( 'THREE.NeuralMaterialLoader: Manifest must be an object.' );

	}

	if ( manifest.format !== FORMAT ) {

		throw new Error( `THREE.NeuralMaterialLoader: Unsupported format "${ manifest.format }" (expected "${ FORMAT }").` );

	}

	if ( manifest.version !== VERSION ) {

		throw new Error( `THREE.NeuralMaterialLoader: Unsupported version ${ manifest.version } (expected ${ VERSION }).` );

	}

	if ( ! manifest.texture || typeof manifest.texture !== 'object' ) {

		throw new Error( 'THREE.NeuralMaterialLoader: Manifest must define a texture block.' );

	}

	if ( ! manifest.channels || ! Array.isArray( manifest.channels.activeKeys ) ) {

		throw new Error( 'THREE.NeuralMaterialLoader: Manifest must define channels.activeKeys.' );

	}

	for ( const key of manifest.channels.activeKeys ) {

		getChannel( key ); // throws a clear error on an unknown channel key

	}

}

// Re-exported for convenience - see `decodeChannelClassification`'s doc
// comment above for why `parse()` doesn't call this itself: a caller that
// wants a loaded material's true semantic constant values (e.g. `Infinity`
// instead of the encoded sentinel, for `attenuationDistance`) for display
// purposes can call `decodeConstantValues( channelClassification.
// constantValues )` on its own copy.
export { NeuralMaterialLoader, decodeConstantValues };

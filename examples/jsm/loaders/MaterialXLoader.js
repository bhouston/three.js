import { FileLoader, Loader } from 'three/webgpu';

import { MaterialXDocument } from './materialx/MaterialXDocument.js';
import { MaterialXIssueCollector } from './materialx/MaterialXWarnings.js';
import { isZipBuffer, readMtlxArchive, createArchiveResolver } from './materialx/MaterialXArchive.js';

const _textDecoder = new TextDecoder();

/**
 * A loader for the MaterialX format.
 *
 * The node materials loaded with this loader can only be used with {@link WebGPURenderer}.
 *
 * ```js
 * const loader = new MaterialXLoader().setPath( SAMPLE_PATH );
 * const { materials, report } = await loader.loadAsync( 'standard_surface_brass_tiled.mtlx' );
 * ```
 *
 * @augments Loader
 * @three_import import { MaterialXLoader } from 'three/addons/loaders/MaterialXLoader.js';
 */
class MaterialXLoader extends Loader {

	/**
	 * Constructs a new MaterialX loader.
	 *
	 * @param {LoadingManager} [manager] - The loading manager.
	 */
	constructor( manager ) {

		super( manager );

		this.unsupportedPolicy = 'warn';
		this.warningCallback = null;
		this.materialName = null;

	}

	/**
	 * Configures how unsupported MaterialX content is handled.
	 *
	 * @param {'warn'|'error'|'ignore'} policy - Unsupported content policy.
	 * @return {MaterialXLoader} A reference to this loader.
	 */
	setUnsupportedPolicy( policy ) {

		this.unsupportedPolicy = policy;
		return this;

	}

	/**
	 * Registers a callback fired whenever the loader reports a warning issue.
	 *
	 * @param {?Function} callback - Warning callback.
	 * @return {MaterialXLoader} A reference to this loader.
	 */
	setWarningCallback( callback ) {

		this.warningCallback = callback;
		return this;

	}

	/**
	 * Selects a specific `surfacematerial` by name.
	 *
	 * @param {?string} materialName - The material name to compile.
	 * @return {MaterialXLoader} A reference to this loader.
	 */
	setMaterialName( materialName ) {

		this.materialName = materialName;
		return this;

	}

	/**
	 * Starts loading from the given URL and passes the loaded MaterialX asset
	 * to the `onLoad()` callback.
	 *
	 * Supports plain `.mtlx` files and `.mtlx.zip` archives that contain a single
	 * `.mtlx` file plus associated textures.
	 *
	 * @param {string} url - The path/URL of the file to be loaded. This can also be a data URI.
	 * @param {function({materials:Object<string,NodeMaterial>,report:Object})} onLoad - Executed when the loading process has been finished.
	 * @param {onProgressCallback} onProgress - Executed while the loading is in progress.
	 * @param {onErrorCallback} onError - Executed when errors occur.
	 * @return {MaterialXLoader} A reference to this loader.
	 */
	load( url, onLoad, onProgress, onError ) {

		const _onError = function ( e ) {

			if ( onError ) {

				onError( e );

			} else {

				console.error( e );

			}

		};

		new FileLoader( this.manager )
			.setPath( this.path )
			.setResponseType( 'arraybuffer' )
			.load( url, ( data ) => {

				try {

					onLoad( this.parseBuffer( data, url ) );

				} catch ( e ) {

					_onError( e );

				}

			}, onProgress, _onError );

		return this;

	}

	/**
	 * Parses loaded bytes and auto-detects ZIP container payloads.
	 *
	 * @param {ArrayBuffer|string|Uint8Array} data - Raw file payload.
	 * @param {string} [url=''] - Optional source URL for extension hinting.
	 * @return {{materials:Object<string,NodeMaterial>,report:Object}} A dictionary holding parsed node materials plus diagnostics.
	 */
	parseBuffer( data, url = '' ) {

		let text;
		let archiveResolver = null;

		if ( data && ( isZipBuffer( data ) || /\.mtlx\.zip$/i.test( url ) ) ) {

			const archive = readMtlxArchive( data );
			text = archive.text;
			archiveResolver = createArchiveResolver( archive.files );

		} else if ( typeof data === 'string' ) {

			text = data;

		} else if ( data instanceof Uint8Array ) {

			text = _textDecoder.decode( data );

		} else {

			text = _textDecoder.decode( new Uint8Array( data ) );

		}

		return this.parse( text, archiveResolver );

	}

	/**
	 * Parses the given MaterialX data and returns resulting materials plus report.
	 *
	 * @param {string} text - The raw MaterialX data as a string.
	 * @param {?Function} archiveResolver - Optional archive URI resolver for in-container textures.
	 * @return {{materials:Object<string,NodeMaterial>,report:Object}} A dictionary holding parse node materials and diagnostics.
	 */
	parse( text, archiveResolver = null ) {

		const issueCollector = new MaterialXIssueCollector( {
			unsupportedPolicy: this.unsupportedPolicy,
			onWarning: this.warningCallback
		} );

		const document = new MaterialXDocument( this.manager, this.path, issueCollector, archiveResolver );
		const result = document.parse( text, this.materialName );

		issueCollector.throwIfNeeded();

		return result;

	}

}

export { MaterialXLoader };

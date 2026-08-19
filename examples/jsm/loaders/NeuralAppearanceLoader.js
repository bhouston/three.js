import {
	ClampToEdgeWrapping,
	DataUtils,
	DataTexture,
	FileLoader,
	HalfFloatType,
	LinearFilter,
	Loader,
	NoColorSpace,
	RGBAFormat,
	RepeatWrapping
} from 'three';
import {
	FORMAT,
	VERSION,
	LATENT_CHANNELS,
	CHANNELS_PER_LEVEL,
	DECODER_INPUT_SIZE as BRDF_INPUT_SIZE,
	IBL_INPUT_SIZE,
	IBL_OUTPUT_SIZE,
	INDIRECT_INPUT_SIZE,
	INDIRECT_OUTPUT_SIZE
} from '../neural-appearance/NeuralAppearanceFormat.js';

/**
 * A loader for compact neural appearance material assets.
 *
 * The format stores a multiresolution latent grid (one RGBA16F texture per
 * level, the same encoding used by neural-texture / neural-material - see
 * NeuralGridModel.js) and a small MLP decoder that evaluates the learned
 * BRDF for a pair of tangent-space directions. Training is intentionally
 * offline; this loader only prepares runtime data for
 * {@link NeuralAppearanceNodeMaterial}.
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

		const levelTextures = manifest.latents.levels.map( ( level, index ) => createLevelTexture( level, `latents.levels[${ index }]`, manifest.latents.wrap ) );

		return {
			isNeuralAppearanceData: true,
			name: manifest.name || '',
			latentTextures: levelTextures,
			levels: levelTextures.length,
			wrap: manifest.latents.wrap || 'repeat',
			outputs: normalizeOutputs( manifest.outputs ),
			referenceEvaluations: manifest.referenceEvaluations || []
		};

	}

}

function createLevelTexture( level, path, wrap ) {

	assertInteger( level.width, `${ path }.width`, 1 );
	assertInteger( level.height, `${ path }.height`, 1 );

	const expectedLength = level.width * level.height * CHANNELS_PER_LEVEL;
	const data = toFloat32Array( level.data, `${ path }.data` );

	if ( data.length !== expectedLength ) {

		throw new Error( `THREE.NeuralAppearanceLoader: ${ path }.data length must be ${ expectedLength }.` );

	}

	const levelWrap = level.wrap || wrap || 'repeat';
	const wrapping = levelWrap === 'repeat' ? RepeatWrapping : ClampToEdgeWrapping;

	const texture = new DataTexture(
		toHalfFloatArray( data ),
		level.width,
		level.height,
		RGBAFormat,
		HalfFloatType,
		undefined,
		wrapping,
		wrapping,
		LinearFilter,
		LinearFilter,
		undefined,
		NoColorSpace
	);

	texture.generateMipmaps = false;
	texture.needsUpdate = true;

	return texture;

}

function toHalfFloatArray( data ) {

	const result = new Uint16Array( data.length );

	for ( let i = 0; i < data.length; i ++ ) {

		result[ i ] = DataUtils.toHalfFloat( data[ i ] );

	}

	return result;

}

function normalizeOutputs( outputs ) {

	if ( ! outputs || ! outputs.ibl || ! outputs.indirectRadiance || ! outputs.indirectIrradiance ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: Manifest must define outputs.ibl, outputs.indirectRadiance, and outputs.indirectIrradiance.' );

	}

	return {
		brdf: normalizeOutputHead( outputs.brdf, 'outputs.brdf', BRDF_INPUT_SIZE, 3, true ),
		ibl: normalizeOutputHead( outputs.ibl, 'outputs.ibl', IBL_INPUT_SIZE, IBL_OUTPUT_SIZE, false ),
		indirectRadiance: normalizeOutputHead( outputs.indirectRadiance, 'outputs.indirectRadiance', INDIRECT_INPUT_SIZE, INDIRECT_OUTPUT_SIZE, false ),
		indirectIrradiance: normalizeOutputHead( outputs.indirectIrradiance, 'outputs.indirectIrradiance', INDIRECT_INPUT_SIZE, INDIRECT_OUTPUT_SIZE, false ),
		emission: outputs.emission ? normalizeOutputHead( outputs.emission, 'outputs.emission', LATENT_CHANNELS, 3, false ) : null,
		opacity: outputs.opacity ? normalizeOpacityHead( outputs.opacity ) : null
	};

}

function normalizeOpacityHead( head ) {

	const opacity = normalizeOutputHead( head, 'outputs.opacity', LATENT_CHANNELS, 1, false );
	const mode = head.mode !== undefined ? head.mode : 'mask';

	if ( mode !== 'mask' && mode !== 'blend' ) {

		throw new Error( `THREE.NeuralAppearanceLoader: Unsupported outputs.opacity.mode "${ mode }".` );

	}

	const alphaCutoff = head.alphaCutoff !== undefined ? head.alphaCutoff : 0.5;

	if ( Number.isFinite( alphaCutoff ) === false || alphaCutoff < 0 || alphaCutoff > 1 ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: outputs.opacity.alphaCutoff must be between zero and one.' );

	}

	opacity.mode = mode;
	opacity.alphaCutoff = alphaCutoff;
	return opacity;

}

function normalizeOutputHead( head, path, expectedInputSize, expectedOutputSize, needsRotation ) {

	if ( ! head || ! Array.isArray( head.layers ) || head.layers.length === 0 ) {

		throw new Error( `THREE.NeuralAppearanceLoader: Manifest must define ${ path }.layers.` );

	}

	assertInteger( head.inputSize, `${ path }.inputSize`, expectedInputSize, expectedInputSize );

	const rotation = head.rotation || null;

	if ( rotation !== null ) {

		assertInteger( rotation.inputSize, `${ path }.rotation.inputSize`, LATENT_CHANNELS, LATENT_CHANNELS );
		assertInteger( rotation.outputSize, `${ path }.rotation.outputSize`, 12, 12 );
		rotation.weights = validateArray( rotation.weights, `${ path }.rotation.weights`, rotation.inputSize * rotation.outputSize );

	} else if ( needsRotation ) {

		throw new Error( `THREE.NeuralAppearanceLoader: ${ path }.rotation is required.` );

	}

	const layers = head.layers.map( ( layer, index ) => {

		const layerPath = `${ path }.layers[${ index }]`;
		assertInteger( layer.inputSize, `${ layerPath }.inputSize`, 1 );
		assertInteger( layer.outputSize, `${ layerPath }.outputSize`, 1 );

		const activation = layer.activation || 'linear';
		if ( activation !== 'linear' && activation !== 'relu' ) {

			throw new Error( `THREE.NeuralAppearanceLoader: Unsupported ${ layerPath }.activation "${ activation }".` );

		}

		return {
			inputSize: layer.inputSize,
			outputSize: layer.outputSize,
			activation,
			weights: validateArray( layer.weights, `${ layerPath }.weights`, layer.inputSize * layer.outputSize ),
			biases: validateArray( layer.biases || [], `${ layerPath }.biases`, layer.outputSize )
		};

	} );

	if ( layers[ 0 ].inputSize !== expectedInputSize ) {

		throw new Error( `THREE.NeuralAppearanceLoader: ${ path }.layers[0].inputSize must be ${ expectedInputSize }.` );

	}

	for ( let i = 1; i < layers.length; i ++ ) {

		if ( layers[ i ].inputSize !== layers[ i - 1 ].outputSize ) {

			throw new Error( `THREE.NeuralAppearanceLoader: ${ path }.layers[${ i }].inputSize does not match the previous output size.` );

		}

	}

	if ( layers[ layers.length - 1 ].outputSize !== expectedOutputSize ) {

		throw new Error( `THREE.NeuralAppearanceLoader: ${ path } final layer outputSize must be ${ expectedOutputSize }.` );

	}

	const outputActivation = head.outputActivation || { type: 'linear' };

	if ( outputActivation.type !== 'linear' && outputActivation.type !== 'exp' && outputActivation.type !== 'scaledSigmoid' && outputActivation.type !== 'sigmoid' ) {

		throw new Error( `THREE.NeuralAppearanceLoader: Unsupported ${ path }.outputActivation.type "${ outputActivation.type }".` );

	}

	return {
		inputSize: head.inputSize,
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

	if ( ! manifest.latents || ! Array.isArray( manifest.latents.levels ) || manifest.latents.levels.length === 0 ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: Manifest must define a non-empty latents.levels array.' );

	}

	if ( manifest.latents.channelsPerLevel !== undefined && manifest.latents.channelsPerLevel !== CHANNELS_PER_LEVEL ) {

		throw new Error( `THREE.NeuralAppearanceLoader: latents.channelsPerLevel must be ${ CHANNELS_PER_LEVEL }.` );

	}

	if ( manifest.latents.levels.length * CHANNELS_PER_LEVEL !== LATENT_CHANNELS ) {

		throw new Error( `THREE.NeuralAppearanceLoader: latents.levels must contain ${ LATENT_CHANNELS / CHANNELS_PER_LEVEL } levels of ${ CHANNELS_PER_LEVEL } channels each.` );

	}

	for ( let levelIndex = 0; levelIndex < manifest.latents.levels.length; levelIndex ++ ) {

		const level = manifest.latents.levels[ levelIndex ];

		if ( level.wrap !== undefined && level.wrap !== 'repeat' && level.wrap !== 'clamp' ) {

			throw new Error( `THREE.NeuralAppearanceLoader: Unsupported latents.levels[${ levelIndex }].wrap "${ level.wrap }".` );

		}

	}

	if ( ! manifest.outputs || ! manifest.outputs.brdf ) {

		throw new Error( 'THREE.NeuralAppearanceLoader: Manifest must define outputs.brdf.' );

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

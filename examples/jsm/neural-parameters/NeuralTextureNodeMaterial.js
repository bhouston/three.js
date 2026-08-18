import * as THREE from 'three';
import { abs, fract, texture, uniformArray, uv, vec3, vec4 } from 'three/tsl';

/**
 * Packs each trained latent grid level into an RGBA float DataTexture so the
 * runtime can rely on ordinary hardware bilinear filtering + repeat wrap
 * addressing for both interpolation and seamless tiling - no manual
 * bilinear/wrap math needed at inference time (unlike the training kernel,
 * which must hand-roll it for the backward pass).
 */
function buildLevelTextures( cpuModel ) {

	return cpuModel.grids.map( ( grid ) => {

		// Half-float, not full float: RGBA32F DataTextures aren't filterable
		// under WebGPU without an opt-in feature, while RGBA16F is filterable
		// by default and gives plenty of precision for latent grid features.
		const data = new Uint16Array( grid.width * grid.height * 4 );

		for ( let p = 0; p < grid.width * grid.height; p ++ ) {

			for ( let c = 0; c < 4; c ++ ) {

				const value = c < grid.channels ? grid.data[ p * grid.channels + c ] : 0;
				data[ p * 4 + c ] = THREE.DataUtils.toHalfFloat( value );

			}

		}

		const levelTexture = new THREE.DataTexture( data, grid.width, grid.height, THREE.RGBAFormat, THREE.HalfFloatType );
		levelTexture.wrapS = THREE.RepeatWrapping;
		levelTexture.wrapT = THREE.RepeatWrapping;
		levelTexture.magFilter = THREE.LinearFilter;
		levelTexture.minFilter = THREE.LinearFilter;
		levelTexture.generateMipmaps = false;
		levelTexture.needsUpdate = true;

		return levelTexture;

	} );

}

/**
 * Builds the TSL expression that evaluates the trained multiresolution grid
 * + MLP decoder at `uvNode`, returning a vec3 color.
 */
function evaluateNeuralTexture( uvNode, cpuModel, levelTextures ) {

	const features = [];

	for ( let i = 0; i < levelTextures.length; i ++ ) {

		const sample = texture( levelTextures[ i ], uvNode );
		const channels = cpuModel.grids[ i ].channels;

		if ( channels > 0 ) features.push( sample.x );
		if ( channels > 1 ) features.push( sample.y );
		if ( channels > 2 ) features.push( sample.z );
		if ( channels > 3 ) features.push( sample.w );

	}

	let activations = features;

	for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

		const layer = cpuModel.decoder.layers[ l ];
		const weightsArray = uniformArray( layer.weights, 'float' );
		const biasesArray = uniformArray( layer.biases, 'float' );
		const next = [];

		for ( let j = 0; j < layer.outputSize; j ++ ) {

			let value = biasesArray.element( j );

			for ( let i = 0; i < layer.inputSize; i ++ ) {

				value = value.add( weightsArray.element( j * layer.inputSize + i ).mul( activations[ i ] ) );

			}

			next.push( layer.activation === 'relu' ? value.max( 0 ) : value );

		}

		activations = next;

	}

	return vec3( activations[ 0 ], activations[ 1 ], activations[ 2 ] );

}

/**
 * Displays the live output of a trained `NeuralTextureTrainer` model: either
 * the raw neural prediction, or (with `mode: 'diff'`) the absolute error
 * against a reference source texture, so both sides of the comparison view
 * can share the same tileable, scale/offset-able UV mapping.
 *
 * @three_import import { NeuralTextureNodeMaterial } from 'three/addons/neural-parameters/NeuralTextureNodeMaterial.js';
 */
class NeuralTextureNodeMaterial extends THREE.NodeMaterial {

	constructor( cpuModel, options = {} ) {

		super();

		this.cpuModel = cpuModel;
		this.lights = false;
		this.toneMapped = false;

		this.levelTextures = buildLevelTextures( cpuModel );

		const uvScaleNode = options.uvScaleNode;
		const uvOffsetNode = options.uvOffsetNode;

		let coord = uv();
		if ( uvScaleNode ) coord = coord.mul( uvScaleNode );
		if ( uvOffsetNode ) coord = coord.add( uvOffsetNode );
		const tiledUV = fract( coord );

		const predicted = evaluateNeuralTexture( tiledUV, cpuModel, this.levelTextures );

		if ( options.mode === 'diff' && options.sourceTexture ) {

			const target = texture( options.sourceTexture, tiledUV ).rgb;
			const errorScale = options.errorScale !== undefined ? options.errorScale : 1;
			this.colorNode = vec4( abs( predicted.sub( target ) ).mul( errorScale ), 1 );

		} else {

			this.colorNode = vec4( predicted, 1 );

		}

	}

	dispose() {

		for ( const levelTexture of this.levelTextures ) levelTexture.dispose();

		super.dispose();

	}

}

export { NeuralTextureNodeMaterial, evaluateNeuralTexture, buildLevelTextures };

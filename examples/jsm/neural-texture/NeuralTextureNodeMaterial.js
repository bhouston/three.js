import * as THREE from 'three';
import { abs, fract, texture, uniformArray, uv, vec3, vec4 } from 'three/tsl';
import {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsVec4,
	packLayerBiasesVec4,
	evaluateLinearLayerVec4
} from '../neural/NeuralMLPTSL.js';
import { createHalfFloatLatentTexture } from '../neural/NeuralHalfFloatTexture.js';
import { triangleWaveEncodeTSL } from '../neural/NeuralGPUComputeTSL.js';

/**
 * Packs each trained latent grid level into an RGBA half-float DataTexture
 * so the runtime can rely on ordinary hardware bilinear filtering + repeat
 * wrap addressing for both interpolation and seamless tiling - no manual
 * bilinear/wrap math needed at inference time (unlike the training kernel,
 * which must hand-roll it for the backward pass).
 */
function buildLevelTextures( cpuModel ) {

	return cpuModel.grids.map( ( grid ) =>
		createHalfFloatLatentTexture( grid.data, grid.width, grid.height, { channels: grid.channels } )
	);

}

/**
 * Builds the TSL expression that evaluates the trained multiresolution grid
 * + MLP decoder at `uvNode`, returning the raw array of `outputChannels`
 * scalar nodes (one per trained channel - callers slice/decode these into
 * whatever physical quantities they represent, see NeuralMaterialFormat.js).
 */
function evaluateNeuralTextureRaw( uvNode, cpuModel, levelTextures ) {

	const features = [];

	for ( let i = 0; i < levelTextures.length; i ++ ) {

		const sample = texture( levelTextures[ i ], uvNode );
		const channels = cpuModel.grids[ i ].channels;

		if ( channels > 0 ) features.push( sample.x );
		if ( channels > 1 ) features.push( sample.y );
		if ( channels > 2 ) features.push( sample.z );
		if ( channels > 3 ) features.push( sample.w );

	}

	// Mirror the training kernel's step 1b: append the same UV-derived input
	// after the grid taps that the model was trained with - either NTC-style
	// tiled positional encoding (see NeuralGridModel.triangleWaveEncode /
	// triangleWaveEncodeTSL) or the raw (u, v) coordinate. Falls back to the
	// pre-`inputEncoding` `peOctaves > 0` check for backward compatibility
	// with any model object not built via `createNeuralTextureModel`.
	const inputEncoding = cpuModel.inputEncoding !== undefined ? cpuModel.inputEncoding : ( cpuModel.peOctaves > 0 ? 'positional' : 'none' );

	if ( inputEncoding === 'positional' ) {

		features.push( ...triangleWaveEncodeTSL( uvNode.x, uvNode.y, cpuModel.peOctaves ) );

	} else if ( inputEncoding === 'raw' ) {

		features.push( uvNode.x, uvNode.y );

	}

	// Shared vec4-packed MLP evaluator (see ../neural/NeuralMLPTSL.js) - same
	// evaluator neural-appearance uses. Packing 4 scalars per vec4 and
	// evaluating each output neuron with TSL.dot() maps to a native GPU
	// dot-product instruction, and evaluateLinearLayerVec4 materializes each
	// layer's output with .toVar() before the next layer consumes it - this
	// used to be a from-scratch, per-scalar-.add() reimplementation here
	// (with its own separately-discovered .toVar() fix for the same WGSL
	// "maximum parser recursive depth" failure the shared version's comment
	// describes) with no principled reason for the two to differ.
	let activations = packVec4Inputs( features );

	for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

		const layer = cpuModel.decoder.layers[ l ];
		const weightsArray = uniformArray( packLayerWeightsVec4( layer.weights, layer.inputSize, layer.outputSize ), 'vec4' );
		const biasesArray = uniformArray( packLayerBiasesVec4( layer.biases ), 'vec4' );
		const inputVectorCount = Math.ceil( layer.inputSize / 4 );

		activations = evaluateLinearLayerVec4(
			activations, layer.inputSize, layer.outputSize, layer.activation,
			( outputIndex, vectorIndex ) => weightsArray.element( outputIndex * inputVectorCount + vectorIndex ),
			( outputVector ) => biasesArray.element( outputVector )
		);

	}

	const lastLayer = cpuModel.decoder.layers[ cpuModel.decoder.layers.length - 1 ];

	return unpackVec4Outputs( activations, lastLayer.outputSize );

}

/**
 * Convenience wrapper for the (outputChannels === 3) single-texture case:
 * evaluates the network and returns just a vec3 color.
 */
function evaluateNeuralTexture( uvNode, cpuModel, levelTextures ) {

	const activations = evaluateNeuralTextureRaw( uvNode, cpuModel, levelTextures );

	return vec3( activations[ 0 ], activations[ 1 ], activations[ 2 ] );

}

/**
 * Displays the live output of a trained `NeuralTextureTrainer` model: either
 * the raw neural prediction, or (with `mode: 'diff'`) the absolute error
 * against a reference source texture, so both sides of the comparison view
 * can share the same tileable, scale/offset-able UV mapping.
 *
 * @three_import import { NeuralTextureNodeMaterial } from 'three/addons/neural-texture/NeuralTextureNodeMaterial.js';
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

export { NeuralTextureNodeMaterial, evaluateNeuralTexture, evaluateNeuralTextureRaw, buildLevelTextures };

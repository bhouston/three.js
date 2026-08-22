import * as THREE from 'three';
import { abs, fract, texture, uv, vec3, vec4 } from 'three/tsl';
import {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4,
	supportsHalfPrecisionStorage,
	createMat4Storage,
	createVec4Storage
} from '../neural/NeuralMLPTSL.js';
import { createHalfFloatLatentTexture } from '../neural/NeuralHalfFloatTexture.js';

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
 *
 * `renderer`, when given (and already `init()`-ed), lets the decoder weights
 * live in a real fp16 storage buffer instead of an fp32 uniform array
 * wherever the backend actually supports it - see NeuralMLPTSL.js's
 * createMat4Storage/createVec4Storage. Omit it (or pass a renderer that
 * doesn't support half storage) to get the original fp32 uniformArray
 * behavior unchanged.
 */
function evaluateNeuralTextureRaw( uvNode, cpuModel, levelTextures, renderer = null ) {

	const features = [];

	for ( let i = 0; i < levelTextures.length; i ++ ) {

		const sample = texture( levelTextures[ i ], uvNode );
		const channels = cpuModel.grids[ i ].channels;

		if ( channels > 0 ) features.push( sample.x );
		if ( channels > 1 ) features.push( sample.y );
		if ( channels > 2 ) features.push( sample.z );
		if ( channels > 3 ) features.push( sample.w );

	}

	// Shared mat4-packed MLP evaluator (see ../neural/NeuralMLPTSL.js) - same
	// evaluator neural-appearance uses. Packing weights into 4x4 blocks and
	// evaluating each layer with a native mat4 * vec4 multiply maps to one
	// hardware FMA-chain instruction per input quad (instead of 4 separate
	// dot() calls, one per output neuron), and evaluateLinearLayerMat4
	// materializes each layer's output with .toVar() before the next layer
	// consumes it - this used to be a from-scratch, per-scalar-.add()
	// reimplementation here (with its own separately-discovered .toVar() fix
	// for the same WGSL "maximum parser recursive depth" failure the shared
	// version's comment describes) with no principled reason for the two to
	// differ.
	const half = supportsHalfPrecisionStorage( renderer );
	let activations = packVec4Inputs( features, half );

	for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

		const layer = cpuModel.decoder.layers[ l ];
		const weights = createMat4Storage( renderer, packLayerWeightsMat4( layer.weights, layer.inputSize, layer.outputSize ) );
		const biases = createVec4Storage( renderer, packLayerBiasesVec4( layer.biases ) );
		const inputVectorCount = Math.ceil( layer.inputSize / 4 );

		activations = evaluateLinearLayerMat4(
			activations, layer.inputSize, layer.outputSize, layer.activation,
			( outputVector, inputVector ) => weights.node.element( outputVector * inputVectorCount + inputVector ),
			( outputVector ) => biases.node.element( outputVector ),
			half
		);

	}

	const lastLayer = cpuModel.decoder.layers[ cpuModel.decoder.layers.length - 1 ];

	return unpackVec4Outputs( activations, lastLayer.outputSize, half );

}

/**
 * Convenience wrapper for the (outputChannels === 3) single-texture case:
 * evaluates the network and returns just a vec3 color.
 */
function evaluateNeuralTexture( uvNode, cpuModel, levelTextures, renderer = null ) {

	const activations = evaluateNeuralTextureRaw( uvNode, cpuModel, levelTextures, renderer );

	return vec3( activations[ 0 ], activations[ 1 ], activations[ 2 ] );

}

/**
 * Displays the live output of a trained `NeuralTextureTrainer` model: either
 * the raw neural prediction, or (with `mode: 'diff'`) the absolute error
 * against a reference source texture, so both sides of the comparison view
 * can share the same tileable, scale/offset-able UV mapping.
 *
 * `options.renderer`, when given (already `init()`-ed), lets the decoder
 * weights use a real fp16 storage buffer instead of an fp32 uniform array
 * on backends that support it - see NeuralMLPTSL.js's createMat4Storage.
 * Omit it to keep the original fp32 uniformArray path unconditionally.
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

		const predicted = evaluateNeuralTexture( tiledUV, cpuModel, this.levelTextures, options.renderer );

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

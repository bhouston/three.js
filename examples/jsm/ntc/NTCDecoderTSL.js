import {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4,
	supportsHalfPrecisionStorage,
	createMat4Storage,
	createVec4Storage
} from './NTCMLPTSL.js';
import { createHalfFloatLatentTexture } from './NTCHalfFloatTexture.js';
import { float, texture } from 'three/tsl';
import { computeLevelLodWeightTSL } from './NTCMipPyramid.js';

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
 * whatever physical quantities they represent, see NTCFormat.js).
 *
 * `renderer`, when given (and already `init()`-ed), lets the decoder weights
 * live in a real fp16 storage buffer instead of an fp32 uniform array
 * wherever the backend actually supports it - see NTCMLPTSL.js's
 * createMat4Storage/createVec4Storage. Omit it to get the original fp32
 * uniformArray behavior unchanged.
 *
 * `lodNode`, when `cpuModel.mipPyramid` is present (see NTCMipPyramid.js /
 * NTCGridPyramidModel.js - `null`/absent for a model trained with
 * `enableMipPyramid: false`, or loaded from a `.ntc` file predating this
 * feature), is the requested LOD (mip index, a TSL float node - e.g. derived
 * from screen-space UV derivatives or an explicit distance-based estimate,
 * see NTCNodeMaterial.js) this decode should reconstruct: every level's
 * sampled feature is scaled by the same LOD fade weight the trainer applied
 * (`computeLevelLodWeightTSL`), and the normalized LOD is concatenated onto
 * the decoder's input, exactly mirroring the training kernel's forward pass
 * (see NTCGPUComputeTSL.js's step 1) - the whole point being that this must
 * match training bit-for-bit, or the decoder sees an input distribution it
 * was never fit against. Defaults to `float(0)` (finest/closest LOD, i.e.
 * every level fully weighted) when `cpuModel.mipPyramid` is present but no
 * `lodNode` was supplied - the same "always full detail" behavior a
 * non-mip-pyramid model has, just paying the (small) extra decoder input
 * width for it. Ignored entirely (no extra input, no per-level fade) when
 * `cpuModel.mipPyramid` is absent, so evaluating a pre-mip-pyramid `.ntc`
 * model is byte-for-byte unchanged.
 */
function evaluateNeuralTextureRaw( uvNode, cpuModel, levelTextures, renderer = null, lodNode = null ) {

	const mipPyramid = cpuModel.mipPyramid || null;
	const resolvedLodNode = mipPyramid ? ( lodNode || float( 0 ) ) : null;

	const features = [];

	for ( let i = 0; i < levelTextures.length; i ++ ) {

		let sample = texture( levelTextures[ i ], uvNode );

		if ( mipPyramid ) {

			const weight = computeLevelLodWeightTSL( resolvedLodNode, mipPyramid.naturalLods[ i ] );
			sample = sample.mul( weight );

		}

		const channels = cpuModel.grids[ i ].channels;

		if ( channels > 0 ) features.push( sample.x );
		if ( channels > 1 ) features.push( sample.y );
		if ( channels > 2 ) features.push( sample.z );
		if ( channels > 3 ) features.push( sample.w );

	}

	// Append the normalized LOD value as the decoder's final input component
	// - must match NTCGridPyramidModel.js's `inputSize = levels * channels +
	// (mipPyramid ? 1 : 0)` / NTCGPUComputeTSL.js's forward pass exactly.
	if ( mipPyramid ) features.push( resolvedLodNode.div( mipPyramid.maxLod ) );

	// Shared mat4-packed MLP evaluator (see NTCMLPTSL.js). Packing weights
	// into 4x4 blocks and evaluating each layer with a native mat4 * vec4
	// multiply maps to one hardware FMA-chain instruction per input quad
	// (instead of 4 separate dot() calls, one per output neuron), and
	// evaluateLinearLayerMat4 materializes each layer's output with .toVar()
	// before the next layer consumes it - see that function's doc comment
	// for the "maximum parser recursive depth" WGSL failure this works
	// around.
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

export { evaluateNeuralTextureRaw, buildLevelTextures };

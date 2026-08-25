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
import { float, int, texture } from 'three/tsl';
import { selectFeatureLevelTSL } from './NTCMipBands.js';

/**
 * Packs each trained latent grid level into an RGBA half-float DataTexture
 * so the runtime can rely on ordinary hardware bilinear filtering + repeat
 * wrap addressing for both interpolation and seamless tiling - no manual
 * bilinear/wrap math needed at inference time (unlike the training kernel,
 * which must hand-roll it for the backward pass). One `DataTexture` per
 * stored feature level - *not* one real GPU mipmap chain (this pyramid's
 * levels each cover a band of several physical mips, via LOD conditioning in
 * the MLP - see NTCMipBands.js - not one level per physical mip, so a real
 * hardware mip chain wouldn't apply here).
 */
function buildLevelTextures( cpuModel ) {

	return cpuModel.grids.map( ( grid ) =>
		createHalfFloatLatentTexture( grid.data, grid.width, grid.height, { channels: grid.channels } )
	);

}

/**
 * Builds the TSL expression that evaluates the trained mip pyramid + MLP
 * decoder at `uvNode`, returning the raw array of `outputChannels` scalar
 * nodes (one per trained channel - callers slice/decode these into whatever
 * physical quantities they represent, see NTCFormat.js).
 *
 * `renderer`, when given (and already `init()`-ed), lets the decoder weights
 * live in a real fp16 storage buffer instead of an fp32 uniform array
 * wherever the backend actually supports it - see NTCMLPTSL.js's
 * createMat4Storage/createVec4Storage. Omit it to get the original fp32
 * uniformArray behavior unchanged.
 *
 * `lodNode` is the requested LOD (mip index, a TSL float node - e.g. derived
 * from screen-space UV derivatives or an explicit distance-based estimate,
 * see NTCNodeMaterial.js) this decode should reconstruct - defaults to
 * `float(0)` (finest/closest LOD) when omitted. It selects exactly one
 * stored grid level (`selectFeatureLevelTSL`, see NTCMipBands.js - the same
 * mapping the training kernel used, see NTCGPUComputeTSL.js's step 1): every
 * other level's hardware-bilinear-sampled tap is multiplied by an exact 0/1
 * selector and summed into a single shared `channels`-wide feature vector,
 * and the normalized LOD is concatenated onto the decoder's input - this
 * must match training bit-for-bit, or the decoder sees an input distribution
 * it was never fit against.
 */
function evaluateNeuralTextureRaw( uvNode, cpuModel, levelTextures, renderer = null, lodNode = null ) {

	const resolvedLodNode = lodNode || float( 0 );
	const selectedLevel = selectFeatureLevelTSL( resolvedLodNode, levelTextures.length, cpuModel.mipsPerLevel );
	const channels = cpuModel.channels;

	// Built as a plain summed expression tree (`.add(...)`), not `.toVar()` +
	// `.addAssign()`: `evaluateNeuralTextureRaw` is called directly while
	// constructing a material's node graph, outside any `Fn()` block, and
	// `.addAssign()` needs a `Fn()` builder stack to record the assignment -
	// used here it silently fails ("No stack defined for assign operation"),
	// leaving every feature at its `.toVar()` initial value (0) regardless of
	// which level's weight was 1, which reads as a UV-independent constant
	// output. A summed `.add()` chain is a pure expression, not a statement,
	// so it doesn't need that stack at all - only the training kernel (which
	// *does* run inside `Fn()`, see NTCGPUComputeTSL.js) uses `.addAssign()`.
	const featureSums = new Array( channels ).fill( null );

	for ( let i = 0; i < levelTextures.length; i ++ ) {

		const sample = texture( levelTextures[ i ], uvNode );
		const weight = selectedLevel.equal( int( i ) ).select( float( 1 ), float( 0 ) );
		const weighted = [ sample.x, sample.y, sample.z, sample.w ].slice( 0, channels ).map( ( c ) => c.mul( weight ) );

		for ( let c = 0; c < channels; c ++ ) featureSums[ c ] = featureSums[ c ] ? featureSums[ c ].add( weighted[ c ] ) : weighted[ c ];

	}

	const features = featureSums;

	// Append the normalized LOD value as the decoder's final input component
	// - must match NTCGridPyramidModel.js's `inputSize = channels + 1` /
	// NTCGPUComputeTSL.js's forward pass exactly.
	// Math.max(1, ...) guards against a genuine maxLod of 0 (a model that
	// only ever supports LOD 0, see NTCGridPyramidModel.js) - dividing by 0
	// there would produce a NaN feature, matching the same guard
	// NTCGPUComputeTSL.js's training kernel already applies.
	features.push( resolvedLodNode.div( Math.max( 1, cpuModel.maxLod ) ) );

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

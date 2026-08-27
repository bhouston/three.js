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
import { buildMipChainTexture } from './NTCHalfFloatTexture.js';
import { float, fract, round, textureLevel, uniformArray } from 'three/tsl';
import { selectFeatureLevelTSL } from './NTCMipBands.js';
import { computePositionalEncodingTSL } from './NTCPositionalEncoding.js';

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
 * `float(0)` (finest/closest LOD) when omitted. It drives one single
 * hardware `textureSampleLevel` call against `mipChainTexture` (built by
 * NTCHalfFloatTexture.js's `buildMipChainTexture`): the GPU brackets
 * `lodNode` between its two nearest physical mip levels and blends them
 * (genuine trilinear - bilinear within each mip, linear between the two),
 * so a fractional LOD - including one that straddles two *different* stored
 * feature levels' bands - reconstructs a smooth cross-fade instead of the
 * old hard 0/1 level-equality switch. The normalized LOD is still
 * concatenated onto the decoder's input exactly as before - this must match
 * training bit-for-bit, or the decoder sees an input distribution it was
 * never fit against.
 */
function evaluateNeuralTextureRaw( uvNode, cpuModel, mipChainTexture, renderer = null, lodNode = null ) {

	const resolvedLodNode = lodNode || float( 0 );
	const channels = cpuModel.channels;

	const sample = textureLevel( mipChainTexture, uvNode, resolvedLodNode );
	const features = [ sample.x, sample.y, sample.z, sample.w ].slice( 0, channels );

	// Append the normalized LOD value as the decoder's final input component
	// - must match NTCGridPyramidModel.js's `inputSize = channels + 1` /
	// NTCGPUComputeTSL.js's forward pass exactly.
	// Math.max(1, ...) guards against a genuine maxLod of 0 (a model that
	// only ever supports LOD 0, see NTCGridPyramidModel.js) - dividing by 0
	// there would produce a NaN feature, matching the same guard
	// NTCGPUComputeTSL.js's training kernel already applies.
	features.push( resolvedLodNode.div( Math.max( 1, cpuModel.maxLod ) ) );

	// Optional positional encoding (see NTCPositionalEncoding.js /
	// NTCGridPyramidModel.js's `positionalEncoding` option) - the selected
	// level's own triangle-wave-encoded fractional texel position, appended
	// right after the LOD value (matching NTCGridPyramidModel.js's
	// `inputSize`).
	//
	// Training (NTCGPUComputeTSL.js) always samples one *exact* integer LOD
	// per training example, never a blend - so the decoder was never fit
	// against a tx/ty blended across two levels either. Since `resolvedLodNode`
	// here can be fractional (hardware trilinear smoothly blends the actual
	// sampled *feature* between two physical mips - see this function's own
	// doc comment), the level used for tx/ty is instead resolved from
	// `round(resolvedLodNode)` - the single nearest physical mip's own stored
	// level - matching training's input distribution as closely as possible
	// rather than inventing a "blended tx/ty" training never produced. This
	// does mean the positional-encoding phase can take a small discontinuous
	// step exactly at a mip transition even though the sampled feature itself
	// blends smoothly through it - an accepted approximation, not a bug (see
	// this repo's plan docs, .cursor/plans/ntc_paper_gap_02_*, for the
	// broader context: this whole feature is an isolated, incremental step
	// that intentionally does not yet change how corners/mips are sampled).
	if ( cpuModel.positionalEncoding ) {

		const roundedLod = round( resolvedLodNode ).clamp( 0, cpuModel.maxLod );
		const selectedLevel = selectFeatureLevelTSL( roundedLod, cpuModel.grids.length, cpuModel.mipsPerLevel );
		// Derived from `cpuModel.grids` (always populated, whether this model
		// came from a live NTCTrainer run or NTCLoader.js) rather than a
		// separate `cpuModel.resolutions` array, which only training-time
		// models carry - grids are always square (width === height, see
		// NTCGridModel.js's createLatentGrid), so either dimension works.
		const levelResolutions = uniformArray( cpuModel.grids.map( ( grid ) => grid.width ), 'float' );
		const resolution = levelResolutions.element( selectedLevel );

		const tx = fract( uvNode.x.mul( resolution ).sub( 0.5 ) );
		const ty = fract( uvNode.y.mul( resolution ).sub( 0.5 ) );

		features.push( ...computePositionalEncodingTSL( tx, ty, cpuModel.positionalEncodingOctaves ) );

	}

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

export { evaluateNeuralTextureRaw, buildMipChainTexture };

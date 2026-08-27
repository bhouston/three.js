import { computeSymmetricScale, packInt8x4, quantizeSymmetricInt8 } from './NTCPackedDotProduct.js';

/**
 * Post-training (not quantization-aware-trained) int8 quantization of the
 * decoder's *hidden* layers, for the optional `dot4I8Packed`-based MLP
 * evaluator (see NTCPackedDotProduct.js / NTCMLPTSL.js's
 * evaluateHiddenLayerDot4I8). The final (always-linear) output layer is
 * never quantized here - see `buildInt8HiddenLayers`'s doc comment.
 *
 * Every value quantized by this module uses a single, *symmetric*,
 * *per-tensor* scale (one scale for an entire weight matrix, one scale for
 * an entire activation vector) - the simplest scheme that still lets a
 * whole `dot4I8Packed` result be dequantized with one multiply
 * (`result * weightScale * inputScale`, see `buildInt8HiddenLayers`'s
 * returned `layers[i].weightScale`/`inputScale`). This is coarser than
 * NTCQuantization.js's per-latent-grid-level, potentially zero-preserving
 * scheme (Section 4.2 of the NVIDIA neural texture compression paper) -
 * appropriate here since this quantizes the *decoder*, not the *latent
 * grids* that scheme targets, and per-tensor is the natural starting point
 * for comparing whether `dot4I8Packed` is worth pursuing further at all
 * (see this repo's plan doc, .cursor/plans/ntc_paper_gap_04_packed_int8_dot_
 * product.plan.md) before investing in anything finer-grained (per-channel
 * scales, or actual quantization-aware training of the decoder itself).
 *
 * Activation *ranges* are derived analytically (interval/worst-case bound
 * propagation through each layer's known weights - see
 * `computeHiddenLayerOutputBound` below), not empirically calibrated by
 * sampling real inputs. This trades some wasted quantization resolution
 * (a bound that's never actually reached by real data "wastes" part of the
 * int8 range) for a hard *correctness* guarantee - every real activation
 * this decoder can ever produce is provably within the bound used to derive
 * its quantization scale, so nothing can silently saturate/clip in a way a
 * calibration sample set happened not to cover.
 */

/**
 * The int8-hidden-layer path's bound on the decoder's raw input vector
 * (grid taps + LOD + optional positional encoding - see
 * NTCGridPyramidModel.js's `inputSize`) - a single scalar covering every
 * input slot (see the module doc comment's "per-tensor" note).
 *
 * The LOD input is always in `[0, 1]` (already normalized - see
 * NTCDecoderTSL.js) and positional encoding terms are always in `[-1, 1]`
 * (triangle waves - see NTCPositionalEncoding.js), so neither can exceed a
 * bound of 1. The grid-tap channels have no such fixed range - trained
 * latent values can grow arbitrarily during optimization - so their bound is
 * the actual current maximum absolute value found by scanning
 * `cpuModel.grids` directly (always available and exact, whether or not this
 * model used QAT - see NTCQuantization.js - unlike relying on a possibly-
 * absent/stale `cpuModel.quantizationRange`).
 */
function computeDecoderInputBound( cpuModel ) {

	let bound = 1; // covers LOD and positional encoding unconditionally

	for ( const grid of cpuModel.grids ) {

		for ( const value of grid.data ) {

			const abs = Math.abs( value );
			if ( abs > bound ) bound = abs;

		}

	}

	return bound;

}

/**
 * Worst-case (over every output neuron) absolute bound on a dense layer's
 * *pre-activation* output `z = W . x + b`, given a single scalar bound on
 * every element of `x` (see the module doc comment's "per-tensor"
 * simplification: every input element is assumed to independently reach
 * `+/-inputBound` in the worst case, an over-approximation but never an
 * under-approximation - the definition of a sound bound). For output neuron
 * `j`: `|z_j| <= |b_j| + inputBound * sum_i |W_ji|` (triangle inequality),
 * and this returns the max of that expression over all `j`.
 */
function computeLayerPreActivationBound( layer, inputBound ) {

	let maxBound = 0;

	for ( let j = 0; j < layer.outputSize; j ++ ) {

		let bound = Math.abs( layer.biases[ j ] );

		for ( let i = 0; i < layer.inputSize; i ++ ) {

			bound += Math.abs( layer.weights[ j * layer.inputSize + i ] ) * inputBound;

		}

		if ( bound > maxBound ) maxBound = bound;

	}

	return maxBound;

}

/**
 * Propagates a pre-activation bound through a hidden-layer activation
 * function to get a bound on its *output* - i.e. what the *next* layer
 * should treat as that value's own input bound.
 *
 * - `'relu'`: `relu(z)` for `|z| <= b` lies in `[0, b]` - `b` itself remains
 *   a valid (if not tight on the negative side) symmetric bound.
 * - `'hgelu'`: NTCMLP.js's hardGELU is *not* monotonic (see its own doc
 *   comment) - it dips to a global minimum of exactly `-0.1875` at `x =
 *   -0.75`, and behaves as the identity for `|x| >= 1.5`. So for a
 *   pre-activation bound `b`: if `b <= 1.5`, the output is confined to
 *   `[hardGELU(-0.75) or the middle-branch value at -b, ...]` - safely
 *   inside `[-0.1875, 1.5]`; if `b > 1.5`, the identity branch dominates and
 *   the output can reach all the way to `+/-b`. Either way, `max(b, 1.5)`
 *   is always a sound (if occasionally loose right at small `b`) symmetric
 *   bound.
 */
function computeActivationOutputBound( preActivationBound, activation ) {

	if ( activation === 'hgelu' ) return Math.max( preActivationBound, 1.5 );

	return preActivationBound; // 'relu' and anything else (treated as unbounded-but-no-worse-than-preActivationBound)

}

/**
 * Quantizes every weight (and, implicitly, the bias - left in fp32, added
 * after dequantization, exactly like NTCMLPTSL.js's evaluateLinearLayerMat4
 * already does) of one hidden layer to int8, packed 4-per-`u32` in the same
 * `[outputNeuron][inputGroup]` layout `evaluateHiddenLayerDot4I8`
 * (NTCMLPTSL.js) expects - `packedWeights[j * groupCount + g]` holds output
 * neuron `j`'s weights for inputs `[4g, 4g+4)`, zero-padded past
 * `layer.inputSize`.
 */
function packLayerWeightsInt8( layer, weightScale ) {

	const groupCount = Math.ceil( layer.inputSize / 4 );
	const packed = new Array( layer.outputSize * groupCount );

	for ( let j = 0; j < layer.outputSize; j ++ ) {

		for ( let g = 0; g < groupCount; g ++ ) {

			const quad = [ 0, 0, 0, 0 ];

			for ( let k = 0; k < 4; k ++ ) {

				const i = g * 4 + k;
				if ( i >= layer.inputSize ) break;

				const weight = layer.weights[ j * layer.inputSize + i ];
				quad[ k ] = quantizeSymmetricInt8( weight, weightScale );

			}

			packed[ j * groupCount + g ] = packInt8x4( quad );

		}

	}

	return packed;

}

/**
 * Builds everything `evaluateHiddenLayerDot4I8` (NTCMLPTSL.js) needs to
 * evaluate every *hidden* layer of `cpuModel.decoder` using
 * `dot4I8Packed` - one entry per hidden layer, in order:
 * `{ inputSize, outputSize, activation, inputScale, weightScale,
 * packedWeights, biases, groupCount }`. The final (always-linear, un-
 * activated) output layer is deliberately excluded - it's typically much
 * narrower (a handful of physical PBR channels) than the hidden layers, so
 * there's little to gain from quantizing it, and its output feeds directly
 * into shading/material properties where int8's coarser precision would be
 * most visible; NTCDecoderTSL.js always evaluates it with the existing
 * fp32/fp16 `evaluateLinearLayerMat4` path regardless of this option.
 *
 * `inputScale`/`weightScale` are each a single symmetric per-tensor scale
 * (see the module doc comment) - `inputScale` covers whatever bound
 * `computeLayerPreActivationBound`'s propagation (seeded from
 * `computeDecoderInputBound`) proved for *this* layer's actual input
 * (the previous layer's real output range, or the raw decoder input for the
 * first hidden layer), and `weightScale` is derived from this layer's own
 * actual trained weight magnitudes (not a bound - the exact max, since
 * unlike activations the weights are already fully known/static at this
 * point).
 */
function buildInt8HiddenLayers( cpuModel ) {

	const layers = cpuModel.decoder.layers;
	const hiddenLayers = layers.slice( 0, - 1 );

	let inputBound = computeDecoderInputBound( cpuModel );
	const result = [];

	for ( const layer of hiddenLayers ) {

		const inputScale = computeSymmetricScale( inputBound );

		let maxAbsWeight = 0;
		for ( const weight of layer.weights ) maxAbsWeight = Math.max( maxAbsWeight, Math.abs( weight ) );
		const weightScale = computeSymmetricScale( maxAbsWeight );

		const groupCount = Math.ceil( layer.inputSize / 4 );
		const packedWeights = packLayerWeightsInt8( layer, weightScale );

		result.push( {
			inputSize: layer.inputSize,
			outputSize: layer.outputSize,
			activation: layer.activation,
			inputScale,
			weightScale,
			packedWeights,
			biases: layer.biases,
			groupCount
		} );

		const preActivationBound = computeLayerPreActivationBound( layer, inputBound );
		inputBound = computeActivationOutputBound( preActivationBound, layer.activation );

	}

	return result;

}

export {
	computeDecoderInputBound,
	computeLayerPreActivationBound,
	computeActivationOutputBound,
	packLayerWeightsInt8,
	buildInt8HiddenLayers
};

import * as THREE from 'three';
import * as TSL from 'three/tsl';

/**
 * Shared TSL-side vec4-packed MLP evaluation, used by both neural-appearance
 * (NeuralAppearanceTSL.js) and neural-texture/neural-material
 * (NeuralTextureNodeMaterial.js's evaluateNeuralTextureRaw). Packing 4
 * scalars per vec4 and evaluating each output neuron's weighted sum with
 * TSL.dot() maps to a native GPU dot-product instruction (one op instead of
 * 4 separate scalar multiply-adds), rather than accumulating with a plain
 * per-scalar `.add()` chain.
 *
 * These two module families used to have separate, independently
 * hand-written evaluators (neural-texture's was scalar-`float`-uniformArray
 * based, not vec4-packed) with no principled reason for the divergence -
 * which meant a correctness/perf fix discovered in one (the `.toVar()`
 * materialization below - see evaluateLinearLayerVec4's comment) never
 * reached the other. Consolidated here so future changes to "how do we
 * safely evaluate an MLP layer in TSL" only have to happen once.
 */

// Packs a flat array of scalar TSL nodes/plain numbers into vec4-grouped TSL
// nodes, zero-padding the final group if `inputs.length` isn't a multiple of
// 4.
function packVec4Inputs( inputs ) {

	const groups = [];
	const groupCount = Math.ceil( inputs.length / 4 );

	for ( let i = 0; i < groupCount; i ++ ) {

		const offset = i * 4;

		groups.push( TSL.vec4(
			inputs[ offset ] ?? 0,
			inputs[ offset + 1 ] ?? 0,
			inputs[ offset + 2 ] ?? 0,
			inputs[ offset + 3 ] ?? 0
		) );

	}

	return groups;

}

// Inverse of packVec4Inputs: extracts `outputSize` scalar nodes back out of
// a vec4-grouped array.
function unpackVec4Outputs( groups, outputSize ) {

	const outputs = [];

	for ( let i = 0; i < outputSize; i ++ ) {

		outputs.push( groups[ Math.floor( i / 4 ) ].element( i % 4 ) );

	}

	return outputs;

}

// CPU-side: packs a flat, row-major `weights[outputSize][inputSize]` array
// into THREE.Vector4s ready for a `uniformArray( ..., 'vec4' )`, laid out as
// `outputIndex * inputVectorCount + vectorIndex` (matching
// evaluateLinearLayerVec4's `getWeightVec4` indexing below), zero-padding
// past `inputSize`/`outputSize`.
function packLayerWeightsVec4( weights, inputSize, outputSize ) {

	const inputVectorCount = Math.ceil( inputSize / 4 );
	const packed = [];

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			const inputBase = vectorIndex * 4;
			const rowOffset = outputIndex * inputSize;

			packed.push( new THREE.Vector4(
				inputBase < inputSize ? ( weights[ rowOffset + inputBase ] || 0 ) : 0,
				inputBase + 1 < inputSize ? ( weights[ rowOffset + inputBase + 1 ] || 0 ) : 0,
				inputBase + 2 < inputSize ? ( weights[ rowOffset + inputBase + 2 ] || 0 ) : 0,
				inputBase + 3 < inputSize ? ( weights[ rowOffset + inputBase + 3 ] || 0 ) : 0
			) );

		}

	}

	return packed;

}

// CPU-side: packs a flat bias array into THREE.Vector4s, zero-padded past
// `biases.length`.
function packLayerBiasesVec4( biases ) {

	const packed = [];
	const vectorCount = Math.ceil( biases.length / 4 );

	for ( let vectorIndex = 0; vectorIndex < vectorCount; vectorIndex ++ ) {

		const offset = vectorIndex * 4;

		packed.push( new THREE.Vector4(
			biases[ offset ] || 0,
			biases[ offset + 1 ] || 0,
			biases[ offset + 2 ] || 0,
			biases[ offset + 3 ] || 0
		) );

	}

	return packed;

}

// Evaluates one fully-connected layer against vec4-packed inputs.
// `getWeightVec4(outputIndex, vectorIndex)` must return the vec4 uniform
// holding weights[outputIndex][vectorIndex*4 .. vectorIndex*4+3] (row-major,
// zero-padded past inputSize) - a closure so callers can source it either
// from one big shared uniform buffer with a per-layer offset
// (neural-appearance, whose GPU training compute shaders separately need
// that combined-buffer layout - see NeuralAppearanceGPUComputeTSL.js) or
// from a plain per-layer uniformArray (neural-texture, which has no such
// constraint). `getBiasVec4(outputVector)` returns the vec4 holding
// biases[outputVector*4 .. outputVector*4+3] (zero-padded past outputSize)
// the same way, or pass `null` for a bias-free layer.
//
// Materializes each output vec4 with `.toVar()` before returning it.
// Without this, each layer's output expression is consumed live by the next
// layer's dot() calls, and for a multi-layer network built inline (not
// inside its own TSL.Fn()) those per-layer expressions compound across
// layers into a single, much larger generated shader - in the worst case
// (several such networks evaluated side by side in one un-Fn'd scope)
// enough to exceed WGSL's private-address-space budget and fail pipeline
// creation outright. See NeuralAppearanceTSL.js's evaluateMLPViaFn for the
// Fn()-scoping half of that fix, needed when a network is evaluated many
// times per pixel in the same shader; this materialization is the other,
// always-needed half.
function evaluateLinearLayerVec4( inputs, inputSize, outputSize, activation, getWeightVec4, getBiasVec4 ) {

	const outputs = [];
	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );

	for ( let outputVector = 0; outputVector < outputVectorCount; outputVector ++ ) {

		const outputBase = outputVector * 4;
		const sums = [ TSL.float( 0 ), TSL.float( 0 ), TSL.float( 0 ), TSL.float( 0 ) ];

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			const inputVector = inputs[ vectorIndex ];

			for ( let component = 0; component < 4; component ++ ) {

				const outputIndex = outputBase + component;

				if ( outputIndex < outputSize ) {

					sums[ component ] = sums[ component ].add( TSL.dot( inputVector, getWeightVec4( outputIndex, vectorIndex ) ) );

				}

			}

		}

		const bias = getBiasVec4 ? getBiasVec4( outputVector ) : TSL.vec4( 0 );
		let value = bias.add( TSL.vec4( sums[ 0 ], sums[ 1 ], sums[ 2 ], sums[ 3 ] ) );

		if ( activation === 'relu' ) value = value.max( 0 );

		outputs.push( value.toVar() );

	}

	return outputs;

}

export { packVec4Inputs, unpackVec4Outputs, packLayerWeightsVec4, packLayerBiasesVec4, evaluateLinearLayerVec4 };

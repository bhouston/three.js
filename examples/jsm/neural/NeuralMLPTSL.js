import * as THREE from 'three';
import * as TSL from 'three/tsl';

/**
 * Shared TSL-side vec4-packed MLP evaluation, used by both neural-appearance
 * (NeuralAppearanceTSL.js) and neural-texture/neural-material
 * (NeuralTextureNodeMaterial.js's evaluateNeuralTextureRaw). Each linear
 * layer's weights are packed into `mat4` uniform blocks (4 outputs x 4
 * inputs per block) and evaluated with a native `mat4 * vec4` multiply -
 * one hardware FMA-chain instruction per input quad, instead of 4 separate
 * `dot(vec4, vec4)` calls (one per output neuron). Same total FLOP count,
 * but far fewer instructions and better register/ALU utilization,
 * particularly on tile-based mobile GPUs (Apple Silicon, Adreno) with
 * dedicated 4-wide vector FMA units.
 *
 * These two module families used to have separate, independently
 * hand-written evaluators (neural-texture's was scalar-`float`-uniformArray
 * based, not vec4-packed) with no principled reason for the divergence -
 * which meant a correctness/perf fix discovered in one (the `.toVar()`
 * materialization below - see evaluateLinearLayerMat4's comment) never
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
// into THREE.Matrix4 blocks ready for a `uniformArray( ..., 'mat4' )`, one
// block per (outputVector, inputVector) quad-pair, laid out as
// `outputVector * inputVectorCount + inputVector` (matching
// evaluateLinearLayerMat4's `getWeightMat4` indexing below). Each block's
// row r holds the weights feeding output `outputVector * 4 + r` from inputs
// `inputVector * 4 .. inputVector * 4 + 3`, zero-padded past
// `inputSize`/`outputSize`. THREE.Matrix4.set() takes its 16 arguments in
// row-major order but stores them column-major internally, so this reads
// naturally as [output][input] here and `mat4Uniform.mul(vec4Input)` on the
// GPU produces the correct matrix-vector product with no transpose
// bookkeeping required at either end.
function packLayerWeightsMat4( weights, inputSize, outputSize ) {

	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );
	const packed = [];

	const weightAt = ( outputIndex, inputIndex ) => {

		if ( outputIndex >= outputSize || inputIndex >= inputSize ) return 0;

		return weights[ outputIndex * inputSize + inputIndex ] || 0;

	};

	for ( let outputVector = 0; outputVector < outputVectorCount; outputVector ++ ) {

		const outputBase = outputVector * 4;

		for ( let inputVector = 0; inputVector < inputVectorCount; inputVector ++ ) {

			const inputBase = inputVector * 4;

			const matrix = new THREE.Matrix4();
			matrix.set(
				weightAt( outputBase, inputBase ), weightAt( outputBase, inputBase + 1 ), weightAt( outputBase, inputBase + 2 ), weightAt( outputBase, inputBase + 3 ),
				weightAt( outputBase + 1, inputBase ), weightAt( outputBase + 1, inputBase + 1 ), weightAt( outputBase + 1, inputBase + 2 ), weightAt( outputBase + 1, inputBase + 3 ),
				weightAt( outputBase + 2, inputBase ), weightAt( outputBase + 2, inputBase + 1 ), weightAt( outputBase + 2, inputBase + 2 ), weightAt( outputBase + 2, inputBase + 3 ),
				weightAt( outputBase + 3, inputBase ), weightAt( outputBase + 3, inputBase + 1 ), weightAt( outputBase + 3, inputBase + 2 ), weightAt( outputBase + 3, inputBase + 3 )
			);

			packed.push( matrix );

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

// Evaluates one fully-connected layer against vec4-packed inputs using
// native `mat4 * vec4` multiplies. `getWeightMat4(outputVector, inputVector)`
// must return the mat4 uniform holding the 4x4 weight block feeding outputs
// `outputVector*4 .. outputVector*4+3` from inputs
// `inputVector*4 .. inputVector*4+3` (see packLayerWeightsMat4 above for the
// exact row/column layout) - a closure so callers can source it either from
// one big shared uniform buffer with a per-layer offset (neural-appearance,
// whose GPU training compute shaders separately need their own flat-buffer
// layout - see NeuralAppearanceGPUComputeTSL.js) or from a plain per-layer
// uniformArray (neural-texture, which has no such constraint). `getBiasVec4
// (outputVector)` returns the vec4 holding
// biases[outputVector*4 .. outputVector*4+3] (zero-padded past outputSize)
// the same way, or pass `null` for a bias-free layer.
//
// Materializes each output vec4 with `.toVar()` before returning it.
// Without this, each layer's output expression is consumed live by the next
// layer's multiply, and for a multi-layer network built inline (not inside
// its own TSL.Fn()) those per-layer expressions compound across layers into
// a single, much larger generated shader - in the worst case (several such
// networks evaluated side by side in one un-Fn'd scope) enough to exceed
// WGSL's private-address-space budget and fail pipeline creation outright.
// See NeuralAppearanceTSL.js's evaluateMLPViaFn for the Fn()-scoping half of
// that fix, needed when a network is evaluated many times per pixel in the
// same shader; this materialization is the other, always-needed half.
function evaluateLinearLayerMat4( inputs, inputSize, outputSize, activation, getWeightMat4, getBiasVec4 ) {

	const outputs = [];
	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );

	for ( let outputVector = 0; outputVector < outputVectorCount; outputVector ++ ) {

		let value = getBiasVec4 ? getBiasVec4( outputVector ) : TSL.vec4( 0 );

		for ( let inputVector = 0; inputVector < inputVectorCount; inputVector ++ ) {

			value = value.add( getWeightMat4( outputVector, inputVector ).mul( inputs[ inputVector ] ) );

		}

		if ( activation === 'relu' ) value = value.max( 0 );

		outputs.push( value.toVar() );

	}

	return outputs;

}

export { packVec4Inputs, unpackVec4Outputs, packLayerWeightsMat4, packLayerBiasesVec4, evaluateLinearLayerMat4 };

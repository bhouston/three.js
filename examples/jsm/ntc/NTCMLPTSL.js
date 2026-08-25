import * as THREE from 'three';
import * as TSL from 'three/tsl';

/**
 * True when `renderer` can back a real fp16 storage buffer
 * (`instancedArray(count, 'hmat4'|'hvec4')`) for MLP weights/biases: the
 * native WebGPU backend (not the WebGL2 fallback, which has no storage
 * buffer concept at all - see NeuralGPUComputeTSL.js) with the `shader-f16`
 * GPU feature actually available. `renderer` may be omitted (callers that
 * don't have one, or haven't finished `await renderer.init()` yet) - this
 * is the only thing that decides fp16 vs the universal fp32 fallback below,
 * so an absent/uninitialized renderer just means "stay on the fallback",
 * never a throw.
 */
function supportsHalfPrecisionStorage( renderer ) {

	return Boolean(
		renderer &&
		renderer.backend &&
		renderer.backend.isWebGPUBackend === true &&
		typeof renderer.hasFeature === 'function' &&
		renderer.hasFeature( 'shader-f16' ) === true
	);

}

// Writes a flat array of THREE.Matrix4 into an `instancedArray(count, 'hmat4')`
// node's packed Uint16Array backing store, using the same
// DataUtils.toHalfFloat() convention as Float16BufferAttribute (see
// TSLHalfPrecision.tests.js). Matrix4.elements is already column-major, the
// same layout WGSL's `array<mat4x4<f16>>` expects per column, so this is a
// direct element-by-element copy - no transpose bookkeeping needed.
function writeMat4HalfStorage( node, matrices ) {

	checkPackedLength( node.value.array.length / 16, matrices );

	const array = node.value.array;

	for ( let m = 0; m < matrices.length; m ++ ) {

		const elements = matrices[ m ].elements;
		const base = m * 16;

		for ( let e = 0; e < 16; e ++ ) {

			array[ base + e ] = THREE.DataUtils.toHalfFloat( elements[ e ] );

		}

	}

	node.value.needsUpdate = true;

}

// Same idea as writeMat4HalfStorage, for an `instancedArray(count, 'hvec4')`
// node backed by a flat array of THREE.Vector4.
function writeVec4HalfStorage( node, vectors ) {

	checkPackedLength( node.value.array.length / 4, vectors );

	const array = node.value.array;

	for ( let v = 0; v < vectors.length; v ++ ) {

		const vector = vectors[ v ];
		const base = v * 4;

		array[ base ] = THREE.DataUtils.toHalfFloat( vector.x );
		array[ base + 1 ] = THREE.DataUtils.toHalfFloat( vector.y );
		array[ base + 2 ] = THREE.DataUtils.toHalfFloat( vector.z );
		array[ base + 3 ] = THREE.DataUtils.toHalfFloat( vector.w );

	}

	node.value.needsUpdate = true;

}

function checkPackedLength( count, next ) {

	if ( next.length !== count ) {

		throw new Error( `THREE.NTCMLPTSL: Packed buffer length mismatch (${ count } !== ${ next.length }).` );

	}

}

function copyMat4ArrayInto( targetArray, matrices ) {

	checkPackedLength( targetArray.length, matrices );

	for ( let i = 0; i < matrices.length; i ++ ) targetArray[ i ].copy( matrices[ i ] );

}

function copyVec4ArrayInto( targetArray, vectors ) {

	checkPackedLength( targetArray.length, vectors );

	for ( let i = 0; i < vectors.length; i ++ ) targetArray[ i ].copy( vectors[ i ] );

}

/**
 * Builds the storage for one packed `mat4`-per-block MLP weight array
 * (see packLayerWeightsMat4), as a `{ node, isHalf, update(matrices) }`
 * bundle: `node.element(i)` reads block `i` exactly like a plain
 * `uniformArray(..., 'mat4')` did before, `isHalf` tells callers whether
 * the vec4 inputs multiplied against it need to be half-typed too (mixing
 * a half mat4 with a float vec4 throws at build time - see
 * TSLHalfPrecision.tests.js), and `update(matrices)` re-uploads new
 * weights in place (used by live training-preview hot-swaps).
 *
 * When `renderer` supports it (see supportsHalfPrecisionStorage above),
 * this is a real fp16 storage buffer - half the bytes on the wire and
 * native fp16 ALU throughput for the matrix-vector multiply. Otherwise it
 * falls back to the original `uniformArray(..., 'mat4')` fp32 uniform
 * buffer, byte-for-byte the previous behavior.
 */
function createMat4Storage( renderer, matrices ) {

	if ( supportsHalfPrecisionStorage( renderer ) ) {

		const node = TSL.instancedArray( matrices.length, 'hmat4' );
		writeMat4HalfStorage( node, matrices );

		return { node, isHalf: true, update: ( next ) => writeMat4HalfStorage( node, next ) };

	}

	const node = TSL.uniformArray( matrices, 'mat4' );

	return { node, isHalf: false, update: ( next ) => copyMat4ArrayInto( node.array, next ) };

}

/**
 * Same as createMat4Storage, for a packed `vec4`-per-block array (see
 * packLayerBiasesVec4) - typically an MLP layer's biases.
 */
function createVec4Storage( renderer, vectors ) {

	if ( supportsHalfPrecisionStorage( renderer ) ) {

		const node = TSL.instancedArray( vectors.length, 'hvec4' );
		writeVec4HalfStorage( node, vectors );

		return { node, isHalf: true, update: ( next ) => writeVec4HalfStorage( node, next ) };

	}

	const node = TSL.uniformArray( vectors, 'vec4' );

	return { node, isHalf: false, update: ( next ) => copyVec4ArrayInto( node.array, next ) };

}

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
// 4. `half: true` (see createMat4Storage/createVec4Storage's `isHalf`) packs
// into `hvec4` groups instead of `vec4` - required at the boundary feeding a
// half-precision weight multiply, since a half mat4 can only be multiplied
// against a half vec4 (see NeuralMLPTSL.js module header / evaluateLinearLayerMat4).
function packVec4Inputs( inputs, half = false ) {

	const vec4Fn = half ? TSL.hvec4 : TSL.vec4;
	const groups = [];
	const groupCount = Math.ceil( inputs.length / 4 );

	for ( let i = 0; i < groupCount; i ++ ) {

		const offset = i * 4;

		groups.push( vec4Fn(
			inputs[ offset ] ?? 0,
			inputs[ offset + 1 ] ?? 0,
			inputs[ offset + 2 ] ?? 0,
			inputs[ offset + 3 ] ?? 0
		) );

	}

	return groups;

}

// Inverse of packVec4Inputs: extracts `outputSize` scalar nodes back out of
// a vec4-grouped array. `half: true` narrows each extracted scalar back to
// `float` (`.toFloat()`) - callers downstream of an MLP's last layer (or an
// intermediate value, like neural-appearance's rotation-frame projection,
// consumed by ordinary float math) always expect float, regardless of what
// precision the weight multiply chain itself ran at.
function unpackVec4Outputs( groups, outputSize, half = false ) {

	const outputs = [];

	for ( let i = 0; i < outputSize; i ++ ) {

		const value = groups[ Math.floor( i / 4 ) ].element( i % 4 );
		outputs.push( half ? value.toFloat() : value );

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
// the same way, or pass `null` for a bias-free layer. `half: true` must
// match whatever precision `inputs`/getWeightMat4/getBiasVec4 actually are
// (see createMat4Storage/createVec4Storage's `isHalf`) - it only picks the
// zero-bias fallback's own type, since half vs float can't mix in a single
// `mat4 * vec4` multiply.
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
function evaluateLinearLayerMat4( inputs, inputSize, outputSize, activation, getWeightMat4, getBiasVec4, half = false ) {

	const outputs = [];
	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );
	const zeroVec4Fn = half ? TSL.hvec4 : TSL.vec4;

	for ( let outputVector = 0; outputVector < outputVectorCount; outputVector ++ ) {

		let value = getBiasVec4 ? getBiasVec4( outputVector ) : zeroVec4Fn( 0 );

		for ( let inputVector = 0; inputVector < inputVectorCount; inputVector ++ ) {

			value = value.add( getWeightMat4( outputVector, inputVector ).mul( inputs[ inputVector ] ) );

		}

		if ( activation === 'relu' ) value = value.max( 0 );

		outputs.push( value.toVar() );

	}

	return outputs;

}

export {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4,
	supportsHalfPrecisionStorage,
	createMat4Storage,
	createVec4Storage
};

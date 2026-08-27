import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Fn, float, storage, uniform } from 'three/tsl';
import { StorageBufferAttribute } from 'three/webgpu';
import { QUANTIZATION_SCHEMES } from '../../../../examples/jsm/ntc/training/NTCQuantization.js';
import { withTestRenderer, evalScalar } from '../helpers/webgpuEval.js';

// QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL is the GPU-side mirror of
// NTCQuantization.js's quantizeForwardCPU - this file cross-checks the two
// agree, and separately guards a real bug this addon's own `bits`/
// `zeroPreserving` implementation hit during development: the
// `zeroPreserving` branch's "zero point" must be a live TSL expression
// derived from the min/max *uniform nodes* themselves, not a JS-side
// constant computed once (e.g. by reading `minNode.value`/`maxNode.value`)
// at kernel-*build* time. `quantization.range: 'auto'` periodically
// refreshes those uniforms' `.value` *without* rebuilding the training
// compute kernel (see NTCTrainer.js's QUANTIZATION_RANGE_REFRESH_INTERVAL) -
// a baked-in zero point would silently go stale the moment the range next
// changed. The last test below reproduces exactly that sequence (build once,
// mutate `.value`, run again without rebuilding) against the same compute
// node object.

describe( 'Addons > NTC > NTCQuantization (real WebGPU)', () => {

	const getRenderer = withTestRenderer( { beforeAll, afterAll } );

	describe( 'quantizeForwardTSL matches quantizeForwardCPU', () => {

		it( 'matches the CPU reference at the default 8 bits, plain (non-zero-preserving) quantization', async () => {

			const renderer = getRenderer();
			const lo = - 2, hi = 3;

			for ( const value of [ - 2, - 0.4, 0, 1.7, 3 ] ) {

				const gpuValue = await evalScalar( renderer, () =>
					QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL( float( value ), float( lo ), float( hi ) )
				);
				const cpuValue = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( value, lo, hi );

				expect( gpuValue ).toBeCloseTo( cpuValue, 4 );

			}

		} );

		it( 'matches the CPU reference at a low bit depth (4 bits)', async () => {

			const renderer = getRenderer();
			const lo = - 2, hi = 3.7, bits = 4;

			for ( const value of [ - 2, - 0.4, 0, 1.7, 3.7 ] ) {

				const gpuValue = await evalScalar( renderer, () =>
					QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL( float( value ), float( lo ), float( hi ), bits )
				);
				const cpuValue = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( value, lo, hi, bits );

				expect( gpuValue ).toBeCloseTo( cpuValue, 4 );

			}

		} );

		it( 'matches the CPU reference with zeroPreserving enabled', async () => {

			const renderer = getRenderer();
			const lo = - 2, hi = 3.7, bits = 4;

			for ( const value of [ - 2, - 0.4, 0, 1.7, 3.7 ] ) {

				const gpuValue = await evalScalar( renderer, () =>
					QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL( float( value ), float( lo ), float( hi ), bits, true )
				);
				const cpuValue = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( value, lo, hi, bits, true );

				expect( gpuValue ).toBeCloseTo( cpuValue, 4 );

			}

		} );

		it( 'zeroPreserving recovers exactly 0 on the GPU too, unlike plain quantization at the same low bit depth', async () => {

			const renderer = getRenderer();
			const lo = - 2, hi = 3.7, bits = 4;

			const plain = await evalScalar( renderer, () =>
				QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL( float( 0 ), float( lo ), float( hi ), bits, false )
			);
			const zeroPreserving = await evalScalar( renderer, () =>
				QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL( float( 0 ), float( lo ), float( hi ), bits, true )
			);

			expect( zeroPreserving ).toBeCloseTo( 0, 5 );
			expect( Math.abs( plain ) ).toBeGreaterThan( 1e-4 );

		} );

	} );

	describe( 'zeroPreserving stays correct when the range uniform changes without rebuilding the kernel', () => {

		it( 'recomputes the zero point live from minNode/maxNode, not a value baked in at kernel-build time', async () => {

			const renderer = getRenderer();
			const bits = 4;

			// Build the compute kernel ONCE, capturing `minUniform`/`maxUniform`
			// as live TSL uniform nodes - exactly how NTCGPUModel.js's
			// `quantizationRangeUniforms` are captured once by
			// createTextureTrainBatchComputeNode and then mutated in place by
			// `setQuantizationRange` on every 'auto' range refresh, without ever
			// rebuilding that compute node.
			const minUniform = uniform( - 2 );
			const maxUniform = uniform( 3.7 );
			const xUniform = uniform( 0 );

			const attribute = new StorageBufferAttribute( new Float32Array( 1 ), 1, Float32Array );
			const out = storage( attribute, 'float', 1 );

			const kernel = Fn( () => {

				out.element( 0 ).assign(
					QUANTIZATION_SCHEMES.uint8.quantizeForwardTSL( xUniform, minUniform, maxUniform, bits, true )
				);

			} )().compute( 1 );

			// First run, at the original range.
			await renderer.computeAsync( kernel );
			const firstBuffer = await renderer.getArrayBufferAsync( attribute );
			const firstResult = new Float32Array( firstBuffer )[ 0 ];
			const expectedFirst = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( 0, - 2, 3.7, bits, true );
			expect( firstResult ).toBeCloseTo( expectedFirst, 4 );

			// Mutate the range uniforms' `.value` in place - no kernel rebuild -
			// exactly like an 'auto' range refresh mid-training.
			minUniform.value = - 5;
			maxUniform.value = 5;

			await renderer.computeAsync( kernel );
			const secondBuffer = await renderer.getArrayBufferAsync( attribute );
			const secondResult = new Float32Array( secondBuffer )[ 0 ];
			const expectedSecond = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( 0, - 5, 5, bits, true );

			// The regression this guards against: if the zero point had been
			// baked in as a JS-side constant from the *original* range at
			// kernel-build time, `secondResult` would still reflect the first
			// range's zero point (and would not match `expectedSecond`, which is
			// computed fresh from the new range) even though the uniforms
			// themselves were updated.
			expect( secondResult ).toBeCloseTo( expectedSecond, 4 );
			expect( secondResult ).toBeCloseTo( 0, 5 ); // both ranges are symmetric about 0

		} );

	} );

} );

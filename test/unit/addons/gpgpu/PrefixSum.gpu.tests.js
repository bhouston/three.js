import { PrefixSum } from '../../../../examples/jsm/gpgpu/PrefixSum.js';
import { isWebGPUAvailable, createRenderer, seededUint32Array } from './gpu-test-utils.js';

// Real-GPU correctness tests for PrefixSum, checked against a plain CPU reference implementation
// of exclusive/inclusive prefix sum. Added after two things turned up while investigating
// PrefixSum/CountingSort GPU performance on this branch:
//
//  1. The subgroup "short spine scan" path (`_getSpineScanShortFn`, selected whenever
//     `numWorkgroups` is small - see `_handleSubgroupInfo`) called a subgroup op from inside
//     branchy control flow, which WGSL disallows in general. The backend rejected the resulting
//     dispatch, logging a GPUValidationError through a channel that turns out not to be
//     observable from page JS in this environment (see the note in ./gpu-test-utils.js) - so a
//     rejected dispatch here shows up as a mismatch against the CPU reference below rather than
//     as a thrown exception.
//  2. Independently, in exclusive mode (`isInclusive: false`) the downsweep pass shifts every
//     write right by one slot; for the last vec4 group that shift computes a write index of
//     `unvectorizedOutputBuffer[count]` - one past the buffer's allocated size. A concrete
//     16-element example: input `[3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3]` produces `[...,68,80]` where
//     the standard exclusive-prefix-sum definition (`output[i] = sum(input[0..i-1])`) gives
//     `[...,68,77]` - the correct value (77) is written first by the in-bounds `.z` component of
//     that last group, then overwritten by the out-of-bounds `.w` component's write landing back
//     on the same index.
//
// Both were found in the same small-`numWorkgroups` code path (a single workgroup covering the
// whole input), which is exactly the shape `CountingSort`'s internal histogram-to-offset prefix
// sum always runs in - its `binCount` is always a power of two, well under these sizes. See
// CountingSort.gpu.tests.js for the same check through the actual `CountingSort` API, including at
// full scale (1,000,000 elements). Whether # 2 affects `CountingSort` in practice depends on
// whether the top bin (`binCount - 1`) is ever actually populated by a given `binNode` - see the
// discussion on the PR this file is attached to.
const SMALL_SIZES = [ 256, 512, 1024, 2048, 4096, 8192 ];

function cpuPrefixSum( input, isInclusive ) {

	const output = new Uint32Array( input.length );
	let running = 0;

	for ( let i = 0; i < input.length; i ++ ) {

		if ( isInclusive ) {

			running += input[ i ];
			output[ i ] = running;

		} else {

			output[ i ] = running;
			running += input[ i ];

		}

	}

	return output;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'PrefixSum (gpu)', () => {

			for ( const n of SMALL_SIZES ) {

				// n is small enough that PrefixSum's partitioning always resolves to a single
				// workgroup (numWorkgroups === 1) regardless of the device's real workgroup-size
				// limit, which is also what selects the subgroup "short spine scan" path.
				QUnit.test( `exclusive prefix sum is correct at n=${ n } (single workgroup / "short" spine-scan path)`, async ( assert ) => {

					assert.timeout( 60000 );

					if ( ! ( await isWebGPUAvailable() ) ) {

						assert.ok( true, 'skipped: WebGPU is not available in this environment' );
						return;

					}

					const renderer = await createRenderer();
					const input = seededUint32Array( n, 10 );
					const expected = cpuPrefixSum( input, false );

					const sum = new PrefixSum( input.slice(), { isInclusive: false } );
					sum.compute( renderer );
					const output = new Uint32Array( await renderer.getArrayBufferAsync( sum.outputAttribute ) );

					assert.deepEqual(
						Array.from( output ), Array.from( expected ),
						'matches a CPU-computed exclusive prefix sum'
					);

					renderer.dispose();

				} );

			}

			QUnit.test( 'inclusive prefix sum is unaffected (isInclusive: true never shifts writes, so it never hits the out-of-bounds write)', async ( assert ) => {

				assert.timeout( 60000 );

				if ( ! ( await isWebGPUAvailable() ) ) {

					assert.ok( true, 'skipped: WebGPU is not available in this environment' );
					return;

				}

				const renderer = await createRenderer();
				const n = 2048;
				const input = seededUint32Array( n, 10 );
				const expected = cpuPrefixSum( input, true );

				const sum = new PrefixSum( input.slice(), { isInclusive: true } );
				sum.compute( renderer );
				const output = new Uint32Array( await renderer.getArrayBufferAsync( sum.outputAttribute ) );

				assert.deepEqual( Array.from( output ), Array.from( expected ), 'matches a CPU-computed inclusive prefix sum' );

				renderer.dispose();

			} );

			QUnit.test( 'exclusive prefix sum is also correct at a large size (multi-workgroup / "long" spine-scan path)', async ( assert ) => {

				assert.timeout( 60000 );

				if ( ! ( await isWebGPUAvailable() ) ) {

					assert.ok( true, 'skipped: WebGPU is not available in this environment' );
					return;

				}

				const renderer = await createRenderer();
				const n = 1_000_000;
				const input = seededUint32Array( n, 10 );
				const expected = cpuPrefixSum( input, false );

				const sum = new PrefixSum( input.slice(), { isInclusive: false } );
				sum.compute( renderer );
				const output = new Uint32Array( await renderer.getArrayBufferAsync( sum.outputAttribute ) );

				assert.deepEqual( Array.from( output ), Array.from( expected ), 'matches a CPU-computed exclusive prefix sum at n=1,000,000' );

				renderer.dispose();

			} );

		} );

	} );

} );

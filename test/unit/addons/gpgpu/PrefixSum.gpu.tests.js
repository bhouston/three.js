import { PrefixSum } from '../../../../examples/jsm/gpgpu/PrefixSum.js';
import { isWebGPUAvailable, createRenderer, seededUint32Array } from './perf-utils.js';

// Regression tests for two real-GPU-only bugs found while profiling PrefixSum/CountingSort (see
// profiling_results.md at the repo root for the full investigation):
//
//  1. The subgroup "short spine scan" path (`_getSpineScanShortFn`, selected whenever
//     `numWorkgroups` is small - see `_handleSubgroupInfo`) called a subgroup op from inside
//     branchy control flow, which WGSL disallows. The backend rejected the dispatch (logging a
//     GPUValidationError through a channel that turns out not to be observable from page JS in
//     this environment - see the note in ./perf-utils.js) instead of throwing in JS, leaving
//     stale/zeroed data behind - which the correctness checks below still catch, since a rejected
//     dispatch never produces the right answer.
//  2. Independently, the downsweep pass's exclusive mode (`isInclusive: false`) shifts every
//     write right by one slot; for the last vec4 group that shift pushes a write to
//     `unvectorizedOutputBuffer[count]` - one past the buffer's real size - silently corrupting
//     the real last element.
//
// Both bugs live in the same small-`numWorkgroups` code path (a single workgroup covering the
// whole input), and `CountingSort`'s internal histogram-to-offset prefix sum always runs in
// exactly that shape - its `binCount` is always a power of two, well under these sizes - which is
// why `CountingSort` always hit both. See CountingSort.gpu.tests.js for the end-to-end
// reproduction through the actual `CountingSort` API.
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

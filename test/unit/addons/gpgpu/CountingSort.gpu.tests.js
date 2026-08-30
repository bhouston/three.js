import { StorageBufferAttribute } from 'three/webgpu';
import { storage, instanceIndex } from 'three/tsl';

import { CountingSort } from '../../../../examples/jsm/gpgpu/CountingSort.js';
import { isWebGPUAvailable, createRenderer, seededUint32Array, captureConsoleErrors } from './perf-utils.js';

// End-to-end regression test through the real `CountingSort` API for the two bugs documented in
// PrefixSum.gpu.tests.js. `CountingSort`'s internal prefix sum (turning its `binCount`-sized
// histogram into per-bin write offsets) always runs in exclusive mode over a power-of-two-sized
// buffer, which is exactly the shape that hit both bugs - so a correct `CountingSort.compute()`
// here is only possible with both fixed.
function assertValidSort( assert, order, keysArray, count ) {

	const seen = new Uint8Array( count );
	let permutationOk = true;

	for ( let i = 0; i < count; i ++ ) {

		const index = order[ i ];

		if ( index >= count || seen[ index ] === 1 ) {

			permutationOk = false;
			break;

		}

		seen[ index ] = 1;

	}

	assert.ok( permutationOk, 'orderAttribute holds a permutation of [0, count) - no index missing or duplicated' );

	let groupedOk = true;
	let lastBin = -1;

	for ( let i = 0; i < count; i ++ ) {

		const bin = keysArray[ order[ i ] ];

		if ( bin < lastBin ) {

			groupedOk = false;
			break;

		}

		lastBin = bin;

	}

	assert.ok( groupedOk, 'the permutation is grouped by ascending bin' );

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'CountingSort (gpu)', () => {

			for ( const [ count, binCount ] of [ [ 5_000, 2048 ], [ 1_000_000, 2048 ] ] ) {

				QUnit.test( `compute() produces a correct, bin-grouped permutation (count=${ count.toLocaleString() }, binCount=${ binCount })`, async ( assert ) => {

					assert.timeout( 60000 );

					if ( ! ( await isWebGPUAvailable() ) ) {

						assert.ok( true, 'skipped: WebGPU is not available in this environment' );
						return;

					}

					const renderer = await createRenderer();
					const keysArray = seededUint32Array( count, binCount );
					const keysRead = storage( new StorageBufferAttribute( keysArray, 1, Uint32Array ), 'uint', count ).toReadOnly();

					const sort = new CountingSort( count, { binCount } );
					sort.setBinNode( () => keysRead.element( instanceIndex ) );

					const { result: order, errors } = await captureConsoleErrors( async () => {

						sort.compute( renderer );
						return new Uint32Array( await renderer.getArrayBufferAsync( sort.orderAttribute ) );

					} );

					assertValidSort( assert, order, keysArray, count );
					assert.deepEqual(
						errors, [],
						'no console.error calls (e.g. a WGSL validation error from CountingSort\'s internal PrefixSum)'
					);

					renderer.dispose();

				} );

			}

		} );

	} );

} );

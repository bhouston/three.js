import { StorageBufferAttribute } from 'three/webgpu';
import { storage, instanceIndex } from 'three/tsl';

import { CountingSort } from '../../../../examples/jsm/gpgpu/CountingSort.js';
import { isWebGPUAvailable, createRenderer, seededUint32Array } from './gpu-test-utils.js';

// End-to-end correctness test through the real `CountingSort` API - see PrefixSum.gpu.tests.js for
// the same checks on `PrefixSum` directly (`CountingSort`'s internal prefix sum, turning its
// `binCount`-sized histogram into per-bin write offsets, always runs in exclusive mode over a
// power-of-two-sized buffer - the same shape flagged there).
//
// `keysArray` is generated across the *full* `[0, binCount)` range on purpose, including the top
// bin, to exercise every bin `CountingSort` could possibly be asked to sort into - whether that
// matches how a given `binNode` is used in practice (i.e. whether the top bin is ever actually
// populated) is a separate question from whether `CountingSort` handles it correctly when it is.
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

					sort.compute( renderer );
					const order = new Uint32Array( await renderer.getArrayBufferAsync( sort.orderAttribute ) );

					assertValidSort( assert, order, keysArray, count );

					renderer.dispose();

				} );

			}

		} );

	} );

} );

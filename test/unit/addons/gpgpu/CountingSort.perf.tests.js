import { StorageBufferAttribute } from 'three/webgpu';
import { storage, instanceIndex } from 'three/tsl';

import { CountingSort } from '../../../../examples/jsm/gpgpu/CountingSort.js';
import { isWebGPUAvailable, createRenderer, benchmark, report } from './perf-utils.js';

// Sizes to benchmark, largest last. Edit this to try other scales.
const SIZES = [ 1_000_000 ];

const BIN_COUNT = 2048;

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'CountingSort (perf)', () => {

			for ( const count of SIZES ) {

				QUnit.test( `compute() over ${ count.toLocaleString() } elements`, async ( assert ) => {

					assert.timeout( 120000 );

					if ( ! ( await isWebGPUAvailable() ) ) {

						assert.ok( true, 'skipped: WebGPU is not available in this environment' );
						return;

					}

					const renderer = await createRenderer();

					// Random bin keys, one per element, read back by `binNode` below.
					const keysArray = new Uint32Array( count );
					for ( let i = 0; i < count; i ++ ) keysArray[ i ] = Math.floor( Math.random() * BIN_COUNT );
					const keysRead = storage( new StorageBufferAttribute( keysArray, 1, Uint32Array ), 'uint', count ).toReadOnly();

					const sort = new CountingSort( count, () => keysRead.element( instanceIndex ), { binCount: BIN_COUNT } );

					const stats = await benchmark(
						() => sort.compute( renderer ),
						() => renderer.getArrayBufferAsync( sort.orderAttribute ),
					);

					report( assert, 'CountingSort.compute()', count, stats );

					renderer.dispose();

				} );

			}

		} );

	} );

} );

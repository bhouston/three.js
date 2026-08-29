import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { BitonicSort } from '../../../../examples/jsm/gpgpu/BitonicSort.js';
import { isWebGPUAvailable, createRenderer, benchmark, report } from './perf-utils.js';

// Sizes to benchmark, largest last. Must be powers of two - BitonicSort's requirement. Edit this
// to try other scales. 2**20 = 1,048,576, the nearest power of two to 1M.
const SIZES = [ 1 << 20 ];

function randomUint32Array( count ) {

	const array = new Uint32Array( count );
	for ( let i = 0; i < count; i ++ ) array[ i ] = Math.floor( Math.random() * 0xffffffff );
	return array;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'BitonicSort (perf)', () => {

			for ( const count of SIZES ) {

				QUnit.test( `compute() over ${ count.toLocaleString() } elements`, async ( assert ) => {

					assert.timeout( 120000 );

					if ( ! ( await isWebGPUAvailable() ) ) {

						assert.ok( true, 'skipped: WebGPU is not available in this environment' );
						return;

					}

					const renderer = await createRenderer();
					const dataAttribute = new StorageBufferAttribute( randomUint32Array( count ), 1, Uint32Array );
					const dataBuffer = storage( dataAttribute, 'uint', count );

					const sort = new BitonicSort( dataBuffer );

					const stats = await benchmark(
						() => sort.compute( renderer ),
						() => renderer.getArrayBufferAsync( dataAttribute ),
					);

					report( assert, 'BitonicSort.compute()', count, stats );

					renderer.dispose();

				} );

			}

		} );

	} );

} );

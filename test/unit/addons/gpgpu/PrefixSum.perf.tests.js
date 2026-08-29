import { PrefixSum } from '../../../../examples/jsm/gpgpu/PrefixSum.js';
import { isWebGPUAvailable, createRenderer, benchmark, report } from './perf-utils.js';

// Sizes to benchmark, largest last. Edit this to try other scales.
const SIZES = [ 1_000_000 ];

function randomUint32Array( count ) {

	const array = new Uint32Array( count );
	for ( let i = 0; i < count; i ++ ) array[ i ] = Math.floor( Math.random() * 1000 );
	return array;

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'PrefixSum (perf)', () => {

			for ( const count of SIZES ) {

				QUnit.test( `compute() over ${ count.toLocaleString() } elements`, async ( assert ) => {

					assert.timeout( 120000 );

					if ( ! ( await isWebGPUAvailable() ) ) {

						assert.ok( true, 'skipped: WebGPU is not available in this environment' );
						return;

					}

					const renderer = await createRenderer();
					const sum = new PrefixSum( randomUint32Array( count ) );

					const stats = await benchmark(
						() => sum.compute( renderer ),
						() => renderer.getArrayBufferAsync( sum.outputAttribute ),
					);

					report( assert, 'PrefixSum.compute()', count, stats );

					renderer.dispose();

				} );

			}

		} );

	} );

} );

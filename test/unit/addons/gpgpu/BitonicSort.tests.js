import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { BitonicSort } from '../../../../examples/jsm/gpgpu/BitonicSort.js';

function createDataBuffer( count ) {

	return storage( new StorageBufferAttribute( count, 1 ), 'uint', count );

}

// A permissive mock renderer: enough for `_ensureBuilt` (and the compute-node graph construction
// it triggers) to run without a real GPU device. `compute()` is a no-op since there is nothing to
// execute it against here.
function createMockRenderer( overrides = {} ) {

	return {
		backend: { device: { limits: {
			maxComputeInvocationsPerWorkgroup: 256,
			maxComputeWorkgroupSizeX: 256,
			maxComputeWorkgroupStorageSize: 16384,
		} } },
		hasFeature: () => false,
		compute: () => {},
		...overrides,
	};

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'BitonicSort', () => {

			QUnit.test( 'count, dispatchSize and swapOpCount are available immediately, without a renderer', ( assert ) => {

				const sort = new BitonicSort( createDataBuffer( 64 ) );

				assert.strictEqual( sort.count, 64 );
				assert.strictEqual( sort.dispatchSize, 32 );
				assert.strictEqual( sort.swapOpCount, 21, 'n=log2(64)=6 -> n*(n+1)/2' );
				assert.strictEqual( sort.workgroupSize, undefined, 'unresolved before the first build' );
				assert.strictEqual( sort.stepCount, undefined, 'depends on workgroupSize, so also unresolved' );

			} );

			QUnit.test( 'workgroupSize is bound by the shared-memory budget and stays a power of two', ( assert ) => {

				const sort = new BitonicSort( createDataBuffer( 64 ) );

				sort._ensureBuilt( createMockRenderer( {
					backend: { device: { limits: {
						maxComputeInvocationsPerWorkgroup: 256,
						maxComputeWorkgroupSizeX: 256,
						maxComputeWorkgroupStorageSize: 32, // 4 bytes/uint * 2 (localStorage holds workgroupSize * 2 elements)
					} } },
				} ) );

				assert.strictEqual( sort.workgroupSize, 4 );
				assert.strictEqual( sort.stepCount, 10 );

			} );

			QUnit.test( 'a full compute() runs stepCount steps and returns to a reset state', ( assert ) => {

				const sort = new BitonicSort( createDataBuffer( 64 ) );
				let dispatches = 0;

				sort.compute( createMockRenderer( {
					backend: { device: { limits: {
						maxComputeInvocationsPerWorkgroup: 256,
						maxComputeWorkgroupSizeX: 256,
						maxComputeWorkgroupStorageSize: 32,
					} } },
					compute: () => {

						dispatches ++;

					},
				} ) );

				assert.strictEqual( sort.stepCount, 10 );
				assert.ok( dispatches >= sort.stepCount, 'at least one renderer.compute() call per step' );
				assert.strictEqual( sort.currentDispatch, 0, 'dispatch counter resets after a full sort' );
				assert.strictEqual( sort.globalOpsRemaining, 0 );
				assert.strictEqual( sort.globalOpsInSpan, 0 );
				assert.strictEqual( sort.readBufferName, 'Data', 'always ends realigned to the original data buffer' );

			} );

			QUnit.test( '_ensureBuilt only builds once, even across renderers with different limits', ( assert ) => {

				const sort = new BitonicSort( createDataBuffer( 16 ) );

				sort._ensureBuilt( createMockRenderer() );
				assert.strictEqual( sort.workgroupSize, 8 );

				sort._ensureBuilt( createMockRenderer( {
					backend: { device: { limits: { maxComputeInvocationsPerWorkgroup: 2, maxComputeWorkgroupSizeX: 2, maxComputeWorkgroupStorageSize: 8 } } },
				} ) );
				assert.strictEqual( sort.workgroupSize, 8, 'the second, more restrictive renderer is ignored once built' );

			} );

		} );

	} );

} );

import { uint } from 'three/tsl';

import { CountingSort } from '../../../../examples/jsm/gpgpu/CountingSort.js';

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

		QUnit.module( 'CountingSort', () => {

			QUnit.test( 'orderRead is available immediately, without a renderer', ( assert ) => {

				const sort = new CountingSort( 8, () => uint( 0 ) );

				assert.ok( sort.orderRead.isStorageBufferNode, 'orderRead is constructed in the constructor' );
				assert.deepEqual( Array.from( sort.orderAttribute.array ), [ 0, 1, 2, 3, 4, 5, 6, 7 ], 'starts as the identity permutation' );

			} );

			QUnit.test( 'binCount scales with count and is clamped to a sane range', ( assert ) => {

				assert.strictEqual( new CountingSort( 10, () => uint( 0 ) ).binCount, 256, 'small counts floor at 256' );
				assert.strictEqual( new CountingSort( 100000, () => uint( 0 ) ).binCount, 2048 );
				assert.strictEqual( new CountingSort( 5000000, () => uint( 0 ) ).binCount, 131072 );

			} );

			QUnit.test( 'binNode is not invoked by construction or by building the node graph', ( assert ) => {

				// TSL only runs a Fn() callback when the node graph is actually compiled to shader
				// code, which needs a real GPU backend -- not available here. This still catches a
				// regression that invokes binNode eagerly (e.g. outside of the Fn() callback).
				let calls = 0;
				const sort = new CountingSort( 8, () => {

					calls ++;
					return uint( 0 );

				} );

				assert.strictEqual( calls, 0, 'constructing the sort does not invoke binNode' );

				// QUnit has no assert.doesNotThrow: an uncaught exception here fails the test on its own.
				sort._ensureBuilt( createMockRenderer() );
				assert.strictEqual( calls, 0, 'building the node graph does not invoke binNode without shader codegen' );

			} );

			QUnit.test( 'workgroupSize is resolved from the renderer on first build', ( assert ) => {

				const sort = new CountingSort( 8, () => uint( 0 ) );

				assert.strictEqual( sort.workgroupSize, undefined, 'unresolved before the first build' );

				sort._ensureBuilt( createMockRenderer( {
					backend: { device: { limits: { maxComputeInvocationsPerWorkgroup: 32, maxComputeWorkgroupSizeX: 256 } } },
				} ) );

				assert.strictEqual( sort.workgroupSize, 32, 'clamped to the device limit' );

			} );

			QUnit.test( 'computeCPU produces a stable permutation grouped by bin', ( assert ) => {

				const bins = [ 2, 0, 1, 0, 2, 1 ];
				const sort = new CountingSort( bins.length, () => uint( 0 ) );
				const versionBefore = sort.orderAttribute.version;

				sort.computeCPU( ( i ) => bins[ i ] );

				assert.deepEqual( Array.from( sort.orderAttribute.array ), [ 1, 3, 2, 5, 0, 4 ] );
				// `needsUpdate` is a write-only setter (see BufferAttribute) that bumps `.version`;
				// reading it back is always `undefined`, so check the version instead.
				assert.ok( sort.orderAttribute.version > versionBefore, 'needsUpdate was set, bumping the version' );

			} );

		} );

	} );

} );

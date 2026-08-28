import { PrefixSum } from '../../../../examples/jsm/gpgpu/PrefixSum.js';

// A permissive mock renderer: enough for `_ensureBuilt` (and the compute-node graph construction
// it triggers) to run without a real GPU device. `hasFeature` reports no subgroup support so
// `_ensureBuilt` takes the single-invocation path, which needs nothing beyond `limits`.
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

		QUnit.module( 'PrefixSum', () => {

			QUnit.test( 'derived sizes are unresolved until compute() first runs', ( assert ) => {

				const sum = new PrefixSum( new Uint32Array( 100 ) );

				assert.strictEqual( sum.workgroupSize, undefined );
				assert.strictEqual( sum.numWorkgroups, undefined );
				assert.strictEqual( sum.dispatchSize, undefined );

			} );

			QUnit.test( 'workgroupSize is the largest power of two the device allows', ( assert ) => {

				const sum = new PrefixSum( new Uint32Array( 100 ) );

				sum.compute( createMockRenderer( {
					backend: { device: { limits: {
						maxComputeInvocationsPerWorkgroup: 100, // not a power of two
						maxComputeWorkgroupSizeX: 256,
						maxComputeWorkgroupStorageSize: 16384,
					} } },
				} ) );

				assert.strictEqual( sum.workgroupSize, 64, 'largest power of two <= 100' );

			} );

			QUnit.test( 'workgroupSize defaults to the full device invocation limit', ( assert ) => {

				const sum = new PrefixSum( new Uint32Array( 100 ) );

				sum.compute( createMockRenderer() );

				// workgroupSize=256, unvectorizedWorkPerInvocation=16 -> partitionSize=4096, so 100 elements fit in one partition.
				assert.strictEqual( sum.workgroupSize, 256, 'nothing in the algorithm favors a smaller workgroup, so it uses the device max' );
				assert.strictEqual( sum.partitionSize, 4096 );
				assert.strictEqual( sum.numWorkgroups, 1 );
				assert.strictEqual( sum.dispatchSize, 256 );

			} );

			QUnit.test( '_ensureBuilt only builds once, even across renderers with different limits', ( assert ) => {

				const sum = new PrefixSum( new Uint32Array( 100 ) );

				sum.compute( createMockRenderer() );
				assert.strictEqual( sum.workgroupSize, 256 );

				sum.compute( createMockRenderer( {
					backend: { device: { limits: { maxComputeInvocationsPerWorkgroup: 8, maxComputeWorkgroupSizeX: 8, maxComputeWorkgroupStorageSize: 16384 } } },
				} ) );
				assert.strictEqual( sum.workgroupSize, 256, 'the second, more restrictive renderer is ignored once built' );

			} );

			QUnit.test( 'isInclusive defaults to true and honors the option', ( assert ) => {

				assert.strictEqual( new PrefixSum( new Uint32Array( 10 ) ).isInclusive, true );
				assert.strictEqual( new PrefixSum( new Uint32Array( 10 ), { isInclusive: false } ).isInclusive, false );

			} );

		} );

	} );

} );

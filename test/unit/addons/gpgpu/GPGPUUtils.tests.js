import { requireLimit, pickWorkgroupSize, pickWorkgroupSizeForSharedMemory } from '../../../../examples/jsm/gpgpu/GPGPUUtils.js';

function createRenderer( limits ) {

	return { backend: { device: { limits } } };

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'GPGPU', () => {

		QUnit.module( 'GPGPUUtils', () => {

			QUnit.test( 'requireLimit reads a named limit off the renderer backend device', ( assert ) => {

				const renderer = createRenderer( { maxComputeInvocationsPerWorkgroup: 256 } );

				assert.strictEqual( requireLimit( renderer, 'maxComputeInvocationsPerWorkgroup' ), 256 );

			} );

			QUnit.test( 'requireLimit throws rather than silently miscomputing when a limit is missing', ( assert ) => {

				// Regression coverage: an `undefined` limit makes every `<=` comparison false, which
				// used to make pickWorkgroupSizeForSharedMemory silently return 1 instead of erroring.
				const renderer = createRenderer( { maxComputeInvocationsPerWorkgroup: 256 } );

				assert.throws( () => requireLimit( renderer, 'maxComputeWorkgroupSizeX' ), /maxComputeWorkgroupSizeX/ );

			} );

			QUnit.test( 'pickWorkgroupSize defaults to 64 clamped by the device limits', ( assert ) => {

				const renderer = createRenderer( { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } );

				assert.strictEqual( pickWorkgroupSize( renderer ), 64, 'uses 64 when the device allows it' );

			} );

			QUnit.test( 'pickWorkgroupSize clamps to the smaller of the two invocation limits', ( assert ) => {

				assert.strictEqual(
					pickWorkgroupSize( createRenderer( { maxComputeInvocationsPerWorkgroup: 32, maxComputeWorkgroupSizeX: 256 } ) ),
					32, 'clamped by maxComputeInvocationsPerWorkgroup'
				);

				assert.strictEqual(
					pickWorkgroupSize( createRenderer( { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 16 } ) ),
					16, 'clamped by maxComputeWorkgroupSizeX'
				);

				assert.strictEqual(
					pickWorkgroupSize( createRenderer( { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } ), 8 ),
					8, 'clamped by the requested preferred size'
				);

			} );

			QUnit.test( 'pickWorkgroupSize throws rather than silently miscomputing when a limit is missing', ( assert ) => {

				assert.throws( () => pickWorkgroupSize( createRenderer( { maxComputeWorkgroupSizeX: 256 } ) ), /maxComputeInvocationsPerWorkgroup/ );
				assert.throws( () => pickWorkgroupSize( createRenderer( { maxComputeInvocationsPerWorkgroup: 256 } ) ), /maxComputeWorkgroupSizeX/ );

			} );

			QUnit.test( 'pickWorkgroupSizeForSharedMemory picks the largest power of two fitting both budgets', ( assert ) => {

				const limits = {
					maxComputeInvocationsPerWorkgroup: 256,
					maxComputeWorkgroupSizeX: 256,
					maxComputeWorkgroupStorageSize: 16384,
				};

				// 4 bytes/invocation: invocation limit (256) binds well before the 16KB shared-memory budget does.
				assert.strictEqual( pickWorkgroupSizeForSharedMemory( createRenderer( limits ), 4 ), 256, 'bound by invocation limit' );

				// Shared-memory budget of 64 bytes at 8 bytes/invocation allows at most 8 invocations.
				assert.strictEqual(
					pickWorkgroupSizeForSharedMemory( createRenderer( { ...limits, maxComputeWorkgroupStorageSize: 64 } ), 8 ),
					8, 'bound by shared-memory budget'
				);

				assert.strictEqual(
					pickWorkgroupSizeForSharedMemory( createRenderer( limits ), 4, 32 ),
					32, 'bound by maxUseful'
				);

			} );

			QUnit.test( 'pickWorkgroupSizeForSharedMemory never returns less than 1', ( assert ) => {

				const renderer = createRenderer( { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupStorageSize: 1 } );

				assert.strictEqual( pickWorkgroupSizeForSharedMemory( renderer, 1024 ), 1 );

			} );

			QUnit.test( 'pickWorkgroupSizeForSharedMemory throws rather than silently miscomputing when a limit is missing', ( assert ) => {

				const renderer = createRenderer( { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } );

				assert.throws( () => pickWorkgroupSizeForSharedMemory( renderer, 4 ), /maxComputeWorkgroupStorageSize/ );

			} );

		} );

	} );

} );

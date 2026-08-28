/**
 * Shared helpers for sizing GPGPU compute shaders ({@link PrefixSum}, {@link CountingSort},
 * {@link BitonicSort}) against a renderer's actual device limits.
 */

/**
 * Reads a named compute limit off `renderer`'s device, throwing if it's missing rather than
 * letting `undefined` silently pass through the arithmetic below.
 *
 * @param {Renderer} renderer
 * @param {string} name - e.g. `'maxComputeInvocationsPerWorkgroup'`.
 * @returns {number}
 */
export function requireLimit( renderer, name ) {

	const value = renderer.backend.device.limits[ name ];

	if ( typeof value !== 'number' ) {

		throw new Error( `GPGPUUtils: renderer.backend.device.limits.${ name } is not a number (got ${ value }).` );

	}

	return value;

}

/**
 * Picks a workgroup size that fits the device's invocation limits.
 *
 * @param {Renderer} renderer
 * @param {number} [preferred=64]
 * @returns {number} `preferred`, clamped down to what the device supports.
 */
export function pickWorkgroupSize( renderer, preferred = 64 ) {

	return Math.min(
		preferred,
		requireLimit( renderer, 'maxComputeInvocationsPerWorkgroup' ),
		requireLimit( renderer, 'maxComputeWorkgroupSizeX' )
	);

}

/**
 * Picks the largest power-of-two workgroup size fitting both the device's invocation limit and a
 * per-invocation workgroup-shared-memory budget.
 *
 * @param {Renderer} renderer
 * @param {number} bytesPerInvocation - Shared-memory bytes used per invocation.
 * @param {number} [maxUseful=Infinity] - Upper bound above which a larger workgroup wouldn't help.
 * @returns {number}
 */
export function pickWorkgroupSizeForSharedMemory( renderer, bytesPerInvocation, maxUseful = Infinity ) {

	const maxInvocations = pickWorkgroupSize( renderer, maxUseful );
	const maxStorageBytes = requireLimit( renderer, 'maxComputeWorkgroupStorageSize' );

	let size = 1;

	while ( size * 2 <= maxInvocations && ( size * 2 ) * bytesPerInvocation <= maxStorageBytes ) {

		size *= 2;

	}

	return size;

}

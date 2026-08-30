// Shared helpers for the GPGPU correctness tests that need a real GPU (the `*.gpu.tests.js` files
// next to this one). Unlike the mocked-renderer correctness tests also next to these files, these
// build a real WebGPURenderer and run actual dispatches, so they only run in a browser with WebGPU
// available - see `test/unit/UnitTestsAddonsGPU.html` and the `test-unit-addons-gpu*` npm scripts.

import { WebGPURenderer } from 'three/webgpu';

// Deliberately not using examples/jsm/capabilities/WebGPU.js here: its top-level `await
// navigator.gpu.requestAdapter()` can race QUnit's autostart (module scripts with a pending
// top-level await don't reliably block it), causing tests to register after the run has already
// ended. Checking availability inside each test body instead - see the `*.gpu.tests.js` files -
// avoids that hazard entirely.

/**
 * @returns {Promise<boolean>} Whether this environment can run the GPU tests at all.
 */
export async function isWebGPUAvailable() {

	if ( typeof navigator === 'undefined' || navigator.gpu === undefined ) return false;

	return Boolean( await navigator.gpu.requestAdapter() );

}

/**
 * A `WebGPUBackend`'s device defaults to `requiredLimits: {}`, which per the WebGPU spec gives the
 * device only the guaranteed-minimum limits (e.g. 256 max compute invocations per workgroup, 16KB
 * max workgroup storage) - not the adapter's real, usually much higher, limits (e.g. 1024/32KB on
 * an Apple M-series GPU). Every workgroup-size decision in the GPGPU addons
 * (`pickWorkgroupSize`/`pickWorkgroupSizeForSharedMemory`, and any bespoke sizing logic in
 * `PrefixSum`/`CountingSort`/`BitonicSort`) reads `renderer.backend.device.limits`, so without this
 * they're all sized against the artificial minimum regardless of what the real device can do -
 * which especially defeats the point of any *device-derived* sizing strategy.
 *
 * @returns {Promise<Object>} A plain object with every limit the adapter actually supports,
 * suitable for `requiredLimits`. (`{ ...adapter.limits }` doesn't work - `GPUSupportedLimits`'
 * properties are getters on its prototype, not its own enumerable properties, so spread/
 * `Object.assign` silently copy nothing; a `for...in` loop is needed to actually read them.)
 */
async function getAdapterLimits() {

	const adapter = await navigator.gpu.requestAdapter();
	const limits = {};

	for ( const key in adapter.limits ) limits[ key ] = adapter.limits[ key ];

	return limits;

}

/**
 * @returns {Promise<WebGPURenderer>} A real, initialized WebGPURenderer, requested with the
 * adapter's real limits (see {@link getAdapterLimits}) rather than the WebGPU spec's
 * guaranteed-minimum defaults.
 */
export async function createRenderer() {

	const renderer = new WebGPURenderer( { requiredLimits: await getAdapterLimits() } );
	await renderer.init();
	return renderer;

}

// A fixed default seed so every GPU test file generates the exact same input data unless told
// otherwise - keeps a test's failure reproducible run to run instead of masking or manufacturing
// a failure depending on which random values happened to come up.
const DEFAULT_SEED = 0x1234abcd;

// mulberry32: a small, fast, deterministic PRNG - good enough for generating test inputs (not for
// anything security-sensitive). Returns a function yielding floats in [0, 1).
function mulberry32( seed ) {

	let a = seed >>> 0;

	return function () {

		a |= 0; a = ( a + 0x6D2B79F5 ) | 0;
		let t = Math.imul( a ^ ( a >>> 15 ), 1 | a );
		t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

/**
 * A deterministically-seeded `Uint32Array` of `count` values in `[0, max)`. Same `count`/`max`/
 * `seed` always produces the same array, on any run.
 */
export function seededUint32Array( count, max, seed = DEFAULT_SEED ) {

	const random = mulberry32( seed );
	const array = new Uint32Array( count );

	for ( let i = 0; i < count; i ++ ) array[ i ] = Math.floor( random() * max );

	return array;

}

// A note on GPU-side validation errors (e.g. a WGSL uniformity violation from an invalid compute
// pipeline): WebGPU reports these to `device.onuncapturederror`/`addEventListener(
// 'uncapturederror', ... )` rather than throwing a catchable JS exception - in principle a good
// fit for asserting "this dispatch didn't just silently get rejected" in a test. In practice,
// neither observably fires in this environment (confirmed empirically: wrapping
// `device.onuncapturederror` - even preserving and forwarding to `WebGPUBackend`'s own handler,
// which already claims that slot to log these itself - stays correctly installed throughout a
// call that we independently know triggers a validation error, and is still never invoked), nor
// does monkey-patching `console.error` (Chrome logs these through an internal, engine-level
// binding that bypasses the page's own `window.console` object entirely). So the GPU correctness
// tests next to this file rely solely on checking actual output values against a CPU reference,
// which is sufficient on its own - a rejected dispatch leaves stale/zeroed data behind rather than
// the right answer, so it fails those checks regardless of whether the error that caused it was
// ever observed. `test/unit/puppeteer.unit.js`'s `--failOnConsoleErrors` flag catches these errors
// as a best-effort supplement, at the puppeteer/CDP level outside the page - see its own comments.

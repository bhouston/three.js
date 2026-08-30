// Shared helpers for the GPGPU performance tests (PrefixSum/CountingSort/BitonicSort). Unlike the
// correctness tests next to these files, these need a real GPU: they build a real WebGPURenderer
// and time actual dispatches, so they only run in a browser with WebGPU available - see
// `test/unit/UnitTestsAddonsPerf.html` and the `test-unit-addons-perf*` npm scripts.

import { WebGPURenderer } from 'three/webgpu';

// Deliberately not using examples/jsm/capabilities/WebGPU.js here: its top-level `await
// navigator.gpu.requestAdapter()` can race QUnit's autostart (module scripts with a pending
// top-level await don't reliably block it), causing tests to register after the run has already
// ended. Checking availability inside each test body instead - see the perf test files - avoids
// that hazard entirely.

/**
 * @returns {Promise<boolean>} Whether this environment can run the perf tests at all.
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

// A fixed default seed so every perf test file, on every branch, generates the exact same input
// data unless told otherwise - that keeps cross-branch timing comparisons apples-to-apples (same
// bin distribution, same comparison outcomes) instead of mixing in run-to-run and branch-to-branch
// data variance on top of the timing variance we actually care about.
const DEFAULT_SEED = 0x1234abcd;

// mulberry32: a small, fast, deterministic PRNG - good enough for generating benchmark inputs
// (not for anything security-sensitive). Returns a function yielding floats in [0, 1).
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
 * `seed` always produces the same array, on any branch or run.
 */
export function seededUint32Array( count, max, seed = DEFAULT_SEED ) {

	const random = mulberry32( seed );
	const array = new Uint32Array( count );

	for ( let i = 0; i < count; i ++ ) array[ i ] = Math.floor( random() * max );

	return array;

}

/**
 * Times repeated runs of `fn`, syncing to the GPU after each one so the measured interval
 * includes actual device execution time, not just CPU-side command submission.
 *
 * @param {Function} fn - Runs one iteration (e.g. `() => sort.compute( renderer )`).
 * @param {Function} sync - Awaited after each `fn()` call to force the GPU work to complete,
 * e.g. `() => renderer.getArrayBufferAsync( someOutputAttribute )`.
 * @param {Object} [options={}]
 * @param {number} [options.runs=20] - Number of timed iterations.
 * @param {number} [options.warmup=3] - Untimed iterations run first, to exclude one-time costs
 * like shader compilation and pipeline creation from the measurements.
 * @returns {Promise<{runs: number, min: number, max: number, mean: number, median: number}>}
 * Timings in milliseconds.
 */
export async function benchmark( fn, sync, { runs = 20, warmup = 3 } = {} ) {

	for ( let i = 0; i < warmup; i ++ ) {

		fn();
		await sync();

	}

	const samples = [];

	for ( let i = 0; i < runs; i ++ ) {

		const start = performance.now();
		fn();
		await sync();
		samples.push( performance.now() - start );

	}

	samples.sort( ( a, b ) => a - b );

	const sum = samples.reduce( ( a, b ) => a + b, 0 );
	const mid = Math.floor( samples.length / 2 );
	const median = ( samples.length % 2 === 0 )
		? ( samples[ mid - 1 ] + samples[ mid ] ) / 2
		: samples[ mid ];

	return { runs, min: samples[ 0 ], max: samples[ samples.length - 1 ], mean: sum / runs, median };

}

/**
 * Logs a formatted timing line to the console and records it as a passing assertion, so it shows
 * up both in the terminal (`puppeteer.unit.js` forwards page console output) and in the QUnit UI
 * when run headful.
 */
export function report( assert, label, count, stats ) {

	const line = `${ label } (count=${ count.toLocaleString() }, n=${ stats.runs } runs): `
		+ `mean=${ stats.mean.toFixed( 3 ) }ms median=${ stats.median.toFixed( 3 ) }ms `
		+ `min=${ stats.min.toFixed( 3 ) }ms max=${ stats.max.toFixed( 3 ) }ms`;

	console.log( line );
	assert.ok( true, line );

}

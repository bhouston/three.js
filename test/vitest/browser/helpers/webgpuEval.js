import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { Fn, storage } from 'three/tsl';

// Minimal "evaluate one TSL node graph, read the number(s) back" harness for
// the vitest browser project (real Chromium + real WebGPU, see
// vitest.config.js). This is deliberately generic - it doesn't know anything
// about neural-appearance/neural-material/neural-texture - so any test file
// that needs to check a TSL node builder's actual numeric output (as opposed
// to just the JS object it returns) can reuse it instead of re-deriving the
// storage()/compute()/getArrayBufferAsync() boilerplate every time.
//
// Why this exists: functions like NeuralOutputActivations.applyChannelActivation
// build a node *graph* (they return a TSL node, not a number) and had no unit
// coverage under the old QUnit harness because QUnit had no way to actually
// run that graph on a GPU and read the result back - every existing test of
// TSL-node-building code either mocks the renderer entirely or skips
// numerical assertions. Here the kernel really runs.

/**
 * Creates and initializes a WebGPURenderer for use within a single test file.
 * Callers are expected to `dispose()` it in an `afterAll` - see
 * `withTestRenderer` below for the common case.
 */
async function createTestRenderer() {

	const renderer = new WebGPURenderer( { forceWebGL: false } );
	await renderer.init();

	return renderer;

}

/**
 * Registers a shared renderer for a whole describe block via vitest's
 * beforeAll/afterAll, and returns a getter for it. One renderer (and one GPU
 * device) per test file keeps things fast without leaking state between
 * files, since each file's tests still run against independently-allocated
 * storage buffers.
 */
function withTestRenderer( { beforeAll, afterAll } ) {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	return () => renderer;

}

/**
 * Runs `buildOutputs(out)` inside a 1-thread compute kernel, where
 * `buildOutputs` assigns `out.element(i)` for `i` in `[0, count)`, then reads
 * the results back as a plain JS number array. Use this for TSL node
 * builders whose output is a single scalar/vector float expression - most of
 * NeuralOutputActivations.js and similar "z -> a" node functions.
 */
async function evalFloats( renderer, count, buildOutputs ) {

	const attribute = new StorageBufferAttribute( new Float32Array( count ), 1, Float32Array );
	const out = storage( attribute, 'float', count );

	const kernel = Fn( () => {

		buildOutputs( out );

	} )().compute( count );

	await renderer.computeAsync( kernel );

	const buffer = await renderer.getArrayBufferAsync( attribute );

	return Array.from( new Float32Array( buffer ) );

}

/**
 * Convenience wrapper around `evalFloats` for a node builder that produces a
 * single scalar, e.g. `evalScalar(renderer, () => applyChannelActivation(float(z), 'sigmoid'))`.
 */
async function evalScalar( renderer, buildNode ) {

	const [ value ] = await evalFloats( renderer, 1, ( out ) => {

		out.element( 0 ).assign( buildNode() );

	} );

	return value;

}

export { createTestRenderer, withTestRenderer, evalFloats, evalScalar };

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { NeuralAppearanceTrainer, createGpuMaterialTeacher } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTrainer.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// Performance instrumentation for NeuralAppearanceTrainer, prompted by the
// user describing neural-appearance training as "quite slow". This isn't a
// pass/fail perf gate (CI hardware varies too much for tight ms budgets to
// be anything but flaky) - it's a *stage breakdown*: wrap the teacher's
// evaluateBatch (every call is a GPU render + a CPU readback - the one
// operation in this pipeline that forces a pipeline stall, see
// NeuralAppearanceTeacherEvaluator.js's own extensive comments about
// grouping/caching evaluateBatch calls to minimize exactly this) and compare
// its cumulative time against total wall-clock, so "is the teacher
// render+readback round trip actually the bottleneck, or is it the GPU
// training compute itself" is an answered, logged question, not a guess.
//
// Bounds are deliberately generous (whole seconds/minutes, not ms) - these
// exist to catch a hang or a gross regression (e.g. an accidental O(n^2) loop
// making a run 100x slower), not to enforce a specific throughput number.
describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceTrainer performance (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	function buildTeacherMaterial() {

		return new THREE.MeshPhysicalNodeMaterial( { color: 0x8899aa, roughness: 0.4, metalness: 0.1 } );

	}

	function buildEnvironmentTexture() {

		// Content doesn't matter for a timing benchmark - just needs to be a
		// valid equirectangular-mapped texture so `createGpuMaterialTeacher`'s
		// `supportsIBL` flips on and the IBL evaluateBatch calls actually run.
		const width = 8, height = 4;
		const data = new Float32Array( width * height * 4 );

		for ( let i = 0; i < width * height; i ++ ) {

			data[ i * 4 + 0 ] = 0.5;
			data[ i * 4 + 1 ] = 0.55;
			data[ i * 4 + 2 ] = 0.6;
			data[ i * 4 + 3 ] = 1;

		}

		const texture = new THREE.DataTexture( data, width, height, THREE.RGBAFormat, THREE.FloatType );
		texture.mapping = THREE.EquirectangularReflectionMapping;
		texture.needsUpdate = true;

		return texture;

	}

	// Instruments any teacher's evaluateBatch calls, returning a getter for
	// cumulative time spent inside them (render + readback), without touching
	// anything else about how the trainer drives the teacher.
	function instrumentTeacher( teacher ) {

		let totalMs = 0;
		let calls = 0;
		const original = teacher.evaluateBatch.bind( teacher );

		teacher.evaluateBatch = async ( ...args ) => {

			const start = performance.now();
			const result = await original( ...args );
			totalMs += performance.now() - start;
			calls ++;
			return result;

		};

		return { getTotalMs: () => totalMs, getCallCount: () => calls };

	}

	it( 'direct-only training (no IBL): reports what fraction of wall-clock is teacher render+readback vs GPU training compute', async () => {

		const material = buildTeacherMaterial();
		const teacher = createGpuMaterialTeacher( material, renderer, { batchSize: 256 } );
		const instrumentation = instrumentTeacher( teacher );

		const trainer = new NeuralAppearanceTrainer( {
			batchSize: 256,
			iterations: 20,
			iblIterations: 0,
			learningRate: 0.001,
			seed: 1
		} );

		const start = performance.now();
		const result = await trainer.train( { material, renderer, teacher } );
		const totalMs = performance.now() - start;
		const teacherMs = instrumentation.getTotalMs();

		console.log(
			`[perf] direct-only: total=${totalMs.toFixed( 0 )}ms teacher=${teacherMs.toFixed( 0 )}ms ` +
			`(${( 100 * teacherMs / totalMs ).toFixed( 1 )}%) other=${( totalMs - teacherMs ).toFixed( 0 )}ms ` +
			`over ${result.iteration} iterations, ${instrumentation.getCallCount()} evaluateBatch calls, ` +
			`final loss=${result.loss.toExponential( 3 )}`
		);

		expect( result.iteration ).toBeGreaterThan( 0 );
		expect( Number.isFinite( result.loss ) ).toBe( true );
		expect( totalMs ).toBeLessThan( 60000 ); // generous smoke bound, not a tight perf gate

	}, 90000 );

	it( 'direct training with IBL enabled: reports the additional teacher-evaluation overhead IBL sampling adds per iteration', async () => {

		const material = buildTeacherMaterial();
		const environment = buildEnvironmentTexture();
		const teacher = createGpuMaterialTeacher( material, renderer, { batchSize: 256, environment } );
		const instrumentation = instrumentTeacher( teacher );

		const trainer = new NeuralAppearanceTrainer( {
			batchSize: 256,
			iterations: 20,
			iblIterations: 0, // isolate the *main-loop* IBL sampling cost, not the dedicated IBL phase
			learningRate: 0.001,
			seed: 1
		} );

		const start = performance.now();
		const result = await trainer.train( { material, renderer, teacher, environment } );
		const totalMs = performance.now() - start;
		const teacherMs = instrumentation.getTotalMs();

		console.log(
			`[perf] direct+IBL: total=${totalMs.toFixed( 0 )}ms teacher=${teacherMs.toFixed( 0 )}ms ` +
			`(${( 100 * teacherMs / totalMs ).toFixed( 1 )}%) other=${( totalMs - teacherMs ).toFixed( 0 )}ms ` +
			`over ${result.iteration} iterations, ${instrumentation.getCallCount()} evaluateBatch calls, ` +
			`final loss=${result.loss.toExponential( 3 )}`
		);

		expect( result.iteration ).toBeGreaterThan( 0 );
		expect( Number.isFinite( result.loss ) ).toBe( true );
		expect( totalMs ).toBeLessThan( 90000 );

		environment.dispose();

	}, 120000 );

} );

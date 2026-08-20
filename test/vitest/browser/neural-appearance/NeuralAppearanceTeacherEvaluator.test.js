import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { vec3 } from 'three/tsl';
import { createGpuMaterialTeacher } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherEvaluator.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// NeuralAppearanceTeacherEvaluator.evaluateBatch() has two layers of caching
// documented in the source (see its own comments above GROUP_BY_MODE,
// _evaluateGrouped and _getModeBundle):
//
//  1. Per-mode *bundle* cache (_modeBundles): the scene/material/render-target
//     for a given targetMode (or merge-group id) is built once and reused for
//     the evaluator's whole lifetime, never rebuilt on a later call.
//  2. Per-batch *result* cache (_lastGroupId/_lastGroupSamples/_lastGroupResults):
//     modes that share an MRT render group (GROUP_BY_MODE) reuse the previous
//     render's results with zero extra render/readback when the immediately
//     preceding evaluateBatch() call rendered the same group for the exact
//     same `samples` array reference.
//
// NeuralAppearanceTrainer.perf.test.js exercises this only indirectly (via
// cumulative timing of real training runs). This file makes both caching
// layers an explicit, direct assertion via vi.spyOn call-count checks on the
// evaluator's own internal methods, instead of inferring behavior from
// wall-clock timing.
describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceTeacherEvaluator (real WebGPU)', () => {

	let renderer;
	const activeTeachers = [];
	const activeEnvironments = [];

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	afterEach( () => {

		for ( const teacher of activeTeachers.splice( 0 ) ) teacher.dispose();
		for ( const environment of activeEnvironments.splice( 0 ) ) environment.dispose();

	} );

	function buildMaterial( { emission = false } = {} ) {

		const material = new THREE.MeshPhysicalNodeMaterial( { color: 0x8899aa, roughness: 0.4, metalness: 0.1 } );
		if ( emission ) material.emissiveNode = vec3( 0.2, 0.1, 0.05 );
		return material;

	}

	function buildEnvironmentTexture() {

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

		activeEnvironments.push( texture );
		return texture;

	}

	function buildSamples( count ) {

		const samples = [];
		for ( let i = 0; i < count; i ++ ) {

			samples.push( {
				uv: [ 0.5, 0.5 ],
				normal: [ 0, 0, 1 ],
				tangent: [ 1, 0, 0 ],
				bitangent: [ 0, 1, 0 ],
				wi: [ 0, 0, 1 ],
				wo: [ 0, 0, 1 ]
			} );

		}

		return samples;

	}

	function createTeacher( material, options = {} ) {

		const teacher = createGpuMaterialTeacher( material, renderer, { teacherBatchSize: 8, ...options } );
		activeTeachers.push( teacher );
		return teacher;

	}

	function expectFiniteVec( vec, size ) {

		expect( vec ).toHaveLength( size );
		for ( const component of vec ) expect( Number.isFinite( component ) ).toBe( true );

	}

	function expectCloseVec( vec, expected, precision = 2 ) {

		expect( vec ).toHaveLength( expected.length );
		for ( let i = 0; i < expected.length; i ++ ) expect( vec[ i ] ).toBeCloseTo( expected[ i ], precision );

	}

	it( 'renders the "direct" MRT group once and serves both brdf and emission from the cached result for the same samples array', async () => {

		const material = buildMaterial( { emission: true } );
		const teacher = createTeacher( material );
		const renderSpy = vi.spyOn( teacher, '_renderAndRead' );

		const samples = buildSamples( 3 );

		const brdf = await teacher.evaluateBatch( samples, 'brdf' );
		expect( renderSpy ).toHaveBeenCalledTimes( 1 );

		const emission = await teacher.evaluateBatch( samples, 'emission' );

		// Same group ('direct'), same samples array reference immediately
		// after -> served from the one-slot cache, no additional render.
		expect( renderSpy ).toHaveBeenCalledTimes( 1 );

		expect( brdf ).toHaveLength( 3 );
		expect( emission ).toHaveLength( 3 );

		// Independent ground truth, not derived from reading the evaluator's
		// own formulas: emissiveNode was set to the constant vec3(0.2, 0.1,
		// 0.05) above, which is unconditional on lighting/geometry/samples --
		// so every sample's emission readout must equal that literal, up to
		// half-float storage precision. This is exactly the kind of thing a
		// caching bug (serving one channel's data for another) would break:
		// if the group cache ever mixed up which attachment belongs to which
		// mode, this would silently start returning the brdf-lit color here
		// instead of the flat emissive constant.
		for ( let i = 0; i < 3; i ++ ) {

			expectFiniteVec( brdf[ i ], 3 );
			expectCloseVec( emission[ i ], [ 0.2, 0.1, 0.05 ], 2 );

		}

	}, 30000 );

	it( 'renders again when the direct group is requested for a different samples array, even with identical content', async () => {

		const material = buildMaterial( { emission: true } );
		const teacher = createTeacher( material );
		const renderSpy = vi.spyOn( teacher, '_renderAndRead' );

		const samplesA = buildSamples( 3 );
		const samplesB = buildSamples( 3 ); // structurally identical, different array reference

		await teacher.evaluateBatch( samplesA, 'brdf' );
		expect( renderSpy ).toHaveBeenCalledTimes( 1 );

		await teacher.evaluateBatch( samplesB, 'emission' );

		// Different `samples` reference breaks the one-slot cache (see
		// _evaluateGrouped's `this._lastGroupSamples === samples` check) ->
		// this must trigger a second render even though the group and the
		// sample content are unchanged.
		expect( renderSpy ).toHaveBeenCalledTimes( 2 );

	}, 30000 );

	it( 'invalidates the cached group result once an interleaving call from a different group runs', async () => {

		const environment = buildEnvironmentTexture();
		const material = buildMaterial( { emission: true } );
		const teacher = createTeacher( material, { environment } );
		const renderSpy = vi.spyOn( teacher, '_renderAndRead' );

		const samples = buildSamples( 3 );

		await teacher.evaluateBatch( samples, 'brdf' ); // group 'direct'
		expect( renderSpy ).toHaveBeenCalledTimes( 1 );

		await teacher.evaluateBatch( samples, 'iblQuery' ); // group 'iblProbe' -- different group, same samples
		expect( renderSpy ).toHaveBeenCalledTimes( 2 );

		// The one-slot cache now holds 'iblProbe', not 'direct' -- asking for
		// 'emission' again (still 'direct') must re-render even though this
		// exact samples array was already used for 'direct' two calls ago.
		await teacher.evaluateBatch( samples, 'emission' );
		expect( renderSpy ).toHaveBeenCalledTimes( 3 );

	}, 30000 );

	it( 'renders the "iblProbe" MRT group once across three evaluateBatch calls for its three merged channels', async () => {

		const environment = buildEnvironmentTexture();
		const material = buildMaterial();
		const teacher = createTeacher( material, { environment } );
		const renderSpy = vi.spyOn( teacher, '_renderAndRead' );

		const samples = buildSamples( 2 );

		const query = await teacher.evaluateBatch( samples, 'iblQuery' );
		const incoming = await teacher.evaluateBatch( samples, 'iblIncoming' );
		const irradiance = await teacher.evaluateBatch( samples, 'iblIrradiance' );

		// iblQuery, iblIncoming and iblIrradiance all share the 'iblProbe'
		// MRT group (GROUP_BY_MODE) -- three evaluateBatch() calls for the
		// same samples array back-to-back must cost exactly one render.
		expect( renderSpy ).toHaveBeenCalledTimes( 1 );

		expect( query ).toHaveLength( 2 );
		expect( incoming ).toHaveLength( 2 );
		expect( irradiance ).toHaveLength( 2 );
		expectFiniteVec( query[ 0 ], 4 );
		expectFiniteVec( incoming[ 0 ], 3 );
		expectFiniteVec( irradiance[ 0 ], 3 );

	}, 30000 );

	it( 'never caches the ungrouped "opacity" mode, even for repeated calls with the identical samples array', async () => {

		const material = buildMaterial();
		const teacher = createTeacher( material );
		const renderSpy = vi.spyOn( teacher, '_renderAndRead' );

		const samples = buildSamples( 3 );

		await teacher.evaluateBatch( samples, 'opacity' );
		expect( renderSpy ).toHaveBeenCalledTimes( 1 );

		await teacher.evaluateBatch( samples, 'opacity' );

		// 'opacity' has no entry in GROUP_BY_MODE, so evaluateBatch() always
		// takes the _evaluateUngrouped() path -- unlike the grouped modes
		// above, an identical repeat call still forces a fresh render.
		expect( renderSpy ).toHaveBeenCalledTimes( 2 );

	}, 30000 );

	it( 'still only builds a mode\'s resource bundle (scene/material/render target) once, independent of the per-call result cache', async () => {

		const material = buildMaterial();
		const teacher = createTeacher( material );
		const resourceSpy = vi.spyOn( teacher, '_createResources' );

		const samplesA = buildSamples( 3 );
		const samplesB = buildSamples( 3 );

		// 'opacity' never hits the per-batch result cache (previous test),
		// but its mode bundle (shader/render-target) must still only be
		// built the first time it's requested.
		await teacher.evaluateBatch( samplesA, 'opacity' );
		await teacher.evaluateBatch( samplesA, 'opacity' );
		await teacher.evaluateBatch( samplesB, 'opacity' );

		expect( resourceSpy ).toHaveBeenCalledTimes( 1 );
		expect( resourceSpy ).toHaveBeenCalledWith( 'opacity' );

	}, 30000 );

	it( 'splits a batch across multiple render calls when samples.length exceeds the configured batch size', async () => {

		const material = buildMaterial();
		const teacher = createTeacher( material, { teacherBatchSize: 4 } );
		const renderSpy = vi.spyOn( teacher, '_renderAndRead' );

		const samples = buildSamples( 10 ); // 4 + 4 + 2 -> three internal batches

		const results = await teacher.evaluateBatch( samples, 'opacity' );

		expect( renderSpy ).toHaveBeenCalledTimes( 3 );
		expect( results ).toHaveLength( 10 );

		// Independent ground truth: this material never sets an opacityNode,
		// and "no opacityNode -> fully opaque" is a well-known, obvious
		// default for three.js materials generally (not something derived
		// from reading this file's formulas) -- so every sample must read
		// back as opaque white (1, 1, 1), regardless of how many internal
		// batches the 10 samples were split across.
		for ( const value of results ) expectCloseVec( value, [ 1, 1, 1 ], 2 );

	}, 30000 );

	it( 'a batch that fits in a single internal chunk costs exactly one render call', async () => {

		const material = buildMaterial();
		const teacher = createTeacher( material, { teacherBatchSize: 4 } );
		const renderSpy = vi.spyOn( teacher, '_renderAndRead' );

		const samples = buildSamples( 4 ); // exactly one internal batch

		const results = await teacher.evaluateBatch( samples, 'opacity' );

		expect( renderSpy ).toHaveBeenCalledTimes( 1 );
		for ( const value of results ) expectCloseVec( value, [ 1, 1, 1 ], 2 );

	}, 30000 );

} );

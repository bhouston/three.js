// CPU-only harness test: simulated time/frame completion, no renderer or GPU.
import assert from 'node:assert/strict';
import { SSGI_PARAMETER_CONFIGS, runSSGIParameterSweep } from './ssgi-parameter-sweep.js';

assert.equal( SSGI_PARAMETER_CONFIGS.length, 15 );
assert.equal( new Set( SSGI_PARAMETER_CONFIGS.map( value => value.label ) ).size, 15 );
assert.equal( new Set( SSGI_PARAMETER_CONFIGS.slice( 0, 9 ).map( value => value.settings.radius ) ).size, 3 );
assert.equal( new Set( SSGI_PARAMETER_CONFIGS.slice( 0, 9 ).map( value => value.settings.thickness ) ).size, 3 );

const performanceDescriptor = Object.getOwnPropertyDescriptor( globalThis, 'performance' );
let clock = 0, referenceCalls = 0, reads = 0, temporalReads = 0, lightingFrozen = false;
Object.defineProperty( globalThis, 'performance', { configurable: true, value: { now: () => clock } } );
const original = { sampling: 'Original', sliceCount: 7, stepCount: 5, radius: 0.9, thickness: 0.02 };
const rawTarget = { width: 4, height: 3 }, filteredTarget = { width: 8, height: 6 };
const state = {
	ssgiParams: { ...original }, params: { output: 'Original Output' }, model: {}, camera: {},
	ssgiPass: { radius: {}, thickness: {}, getRenderTarget: () => rawTarget },
	ssgiDenoise: { _frames: 0, getRenderTarget: () => filteredTarget },
	lightingHistory: { width: 8, height: 6 },
	updateSSGIParameters() {

		this.ssgiPass.radius.value = this.ssgiParams.radius * 10;
		this.ssgiPass.thickness.value = this.ssgiParams.thickness * 10;
		this.ssgiDenoise._frames = 0;

	},
	updateOutput() {}
};
try {

	const results = await runSSGIParameterSweep( {
		state,
		step: () => { clock += 50; state.ssgiDenoise._frames ++; },
		drain: async () => {},
		read: async () => {

			reads ++;
			if ( lightingFrozen ) {

				temporalReads ++;
				return new Float32Array( [ 2, 2, 2, temporalReads % 2 ? 0.5 : 1 ] );

			}
			return new Float32Array( 4 );

		},
		freezeLighting: value => { lightingFrozen = value; },
		getFreezeLighting: () => lightingFrozen,
		geometryReference: async options => {

			referenceCalls ++;
			assert.equal( reads, ( referenceCalls - 1 ) * 35 + 3, 'Geometry follows its three source readbacks; prior temporal reads are outside the next timing interval' );
			assert.equal( lightingFrozen, false, 'Source freeze is restored before the next measured implementation' );
			assert.equal( options.radius, state.ssgiParams.radius * 10 );
			assert.equal( options.raw.width, 4 );
			assert.equal( options.filtered.width, 8 );
			assert.equal( state.params.output, 'Beauty' );
			return { valid: [ true, false, true ], coverage: [ 0.5, 1, 0.25 ], gridWidth: 3, gridHeight: 1, rays: 128, comparisons: {}, method: 'injected CPU harness test', milliseconds: 1 };

		}
	} );
	assert.equal( results.length, 15 );
	assert.equal( referenceCalls, 15 );
	assert.equal( temporalReads, 15 * 32 );
	assert.ok( results.every( value => value.temporal.frames === 32 && value.temporal.sourceFreezeControlled ) );
	assert.ok( results.every( value => Math.abs( value.temporal.temporalDeviation - 0.5 ) < 1e-12 && Math.abs( value.temporal.coverageDeviation - 0.25 ) < 1e-12 ) );
	assert.ok( results.every( value => value.temporal.meanCoverage === 0.75 && value.temporal.meanPremultipliedRGB.every( channel => channel === 1.5 ) ) );
	assert.equal( lightingFrozen, false );
	assert.ok( results.every( value => value.throughput.seconds === 5 && value.throughput.completed === 100 && value.throughput.giFrames === 100 ) );
	assert.ok( results.every( value => value.geometry.validCount === 2 && value.geometry.meanTrueCoverage === 0.375 ) );
	assert.equal( results.find( value => value.label === 'history-8' ).settings.maxFrames, 8 );
	assert.equal( results.find( value => value.label === 'denoise-radius-1' ).settings.maxFrames, 16, 'Each quality case restores the baseline' );
	assert.deepEqual( state.ssgiParams, original, 'All original settings are restored without leaving introduced keys' );
	assert.equal( state.params.output, 'Original Output' );
	lightingFrozen = true;
	await assert.rejects( runSSGIParameterSweep( {
		state,
		step: () => { clock += 50; state.ssgiDenoise._frames ++; },
		drain: async () => {},
		read: async () => new Float32Array( 4 ),
		geometryReference: async () => { throw new Error( 'injected reference failure' ); },
		freezeLighting: value => { lightingFrozen = value; },
		getFreezeLighting: () => lightingFrozen
	} ), /injected reference failure/ );
	assert.deepEqual( state.ssgiParams, original, 'Reference errors also restore all original settings' );
	assert.equal( lightingFrozen, true, 'An initially frozen source stays frozen after failure' );
	assert.equal( state.params.output, 'Original Output' );

	lightingFrozen = false;
	await assert.rejects( runSSGIParameterSweep( {
		state,
		step: () => { clock += 50; state.ssgiDenoise._frames ++; },
		drain: async () => {},
		read: async () => {

			if ( lightingFrozen ) throw new Error( 'injected temporal readback failure' );

			return new Float32Array( 4 );

		},
		geometryReference: async () => ( { valid: [ true ], coverage: [ 0 ], comparisons: {} } ),
		freezeLighting: value => { lightingFrozen = value; },
		getFreezeLighting: () => lightingFrozen
	} ), /injected temporal readback failure/ );
	assert.equal( lightingFrozen, false, 'Temporal readback failures restore source copying' );
	assert.deepEqual( state.ssgiParams, original );

	let mutationReadbacks = 0;
	await assert.rejects( runSSGIParameterSweep( {
		state,
		step: () => {

			clock += 50;
			state.ssgiDenoise._frames ++;
			state.ssgiParams.thickness = 0.0001;

		},
		drain: async () => {},
		read: async () => { mutationReadbacks ++; return new Float32Array( 4 ); },
		geometryReference: async () => { throw new Error( 'Reference must not run after parameter mutation' ); }
	} ), /Parameter mutation in radius-0\.1-thickness-0\.0025: thickness: requested 0\.0025, observed 0\.0001/ );
	assert.equal( mutationReadbacks, 0, 'External slider mutation fails before throughput readback or reference work' );
	assert.deepEqual( state.ssgiParams, original, 'Mutation failures restore caller settings' );

} finally {

	Object.defineProperty( globalThis, 'performance', performanceDescriptor );

}

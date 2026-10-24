// CPU-only harness test: simulated time/frame completion, no renderer or GPU.
import assert from 'node:assert/strict';
import { SSGI_PARAMETER_CONFIGS, runSSGIParameterSweep } from './ssgi-parameter-sweep.js';

assert.equal( SSGI_PARAMETER_CONFIGS.length, 15 );
assert.equal( new Set( SSGI_PARAMETER_CONFIGS.map( value => value.label ) ).size, 15 );
assert.equal( new Set( SSGI_PARAMETER_CONFIGS.slice( 0, 9 ).map( value => value.settings.radius ) ).size, 3 );
assert.equal( new Set( SSGI_PARAMETER_CONFIGS.slice( 0, 9 ).map( value => value.settings.thickness ) ).size, 3 );

const performanceDescriptor = Object.getOwnPropertyDescriptor( globalThis, 'performance' );
let clock = 0, referenceCalls = 0, reads = 0;
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
		read: async () => { reads ++; return new Float32Array( 4 ); },
		geometryReference: async options => {

			referenceCalls ++;
			assert.equal( reads, referenceCalls * 3, 'All GPU readbacks precede CPU geometry work' );
			assert.equal( options.radius, state.ssgiParams.radius * 10 );
			assert.equal( options.raw.width, 4 );
			assert.equal( options.filtered.width, 8 );
			assert.equal( state.params.output, 'Beauty' );
			return { valid: [ true, false, true ], coverage: [ 0.5, 1, 0.25 ], gridWidth: 3, gridHeight: 1, rays: 128, comparisons: {}, method: 'injected CPU harness test', milliseconds: 1 };

		}
	} );
	assert.equal( results.length, 15 );
	assert.equal( referenceCalls, 15 );
	assert.ok( results.every( value => value.throughput.seconds === 5 && value.throughput.completed === 100 && value.throughput.giFrames === 100 ) );
	assert.ok( results.every( value => value.geometry.validCount === 2 && value.geometry.meanTrueCoverage === 0.375 ) );
	assert.equal( results.find( value => value.label === 'history-8' ).settings.maxFrames, 8 );
	assert.equal( results.find( value => value.label === 'denoise-radius-1' ).settings.maxFrames, 16, 'Each quality case restores the baseline' );
	assert.deepEqual( state.ssgiParams, original, 'All original settings are restored without leaving introduced keys' );
	assert.equal( state.params.output, 'Original Output' );
	await assert.rejects( runSSGIParameterSweep( {
		state,
		step: () => { clock += 50; state.ssgiDenoise._frames ++; },
		drain: async () => {},
		read: async () => new Float32Array( 4 ),
		geometryReference: async () => { throw new Error( 'injected reference failure' ); }
	} ), /injected reference failure/ );
	assert.deepEqual( state.ssgiParams, original, 'Reference errors also restore all original settings' );
	assert.equal( state.params.output, 'Original Output' );

} finally {

	Object.defineProperty( globalThis, 'performance', performanceDescriptor );

}

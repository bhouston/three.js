import { describe, expect, it } from 'vitest';

// Proves the vitest browser project (Chromium via the playwright provider,
// see vitest.config.js) actually has real WebGPU available before anything
// else in test/vitest/browser/ relies on it. If this fails, every other
// browser-project test is failing for the same infrastructure reason, not
// because of a bug in the module under test - check this file first.
describe( 'WebGPU availability (infra smoke test)', () => {

	it( 'exposes navigator.gpu in a secure context', () => {

		expect( 'gpu' in navigator ).toBe( true );
		expect( window.isSecureContext ).toBe( true );

	} );

	it( 'can request a real adapter and device', async () => {

		const adapter = await navigator.gpu.requestAdapter();
		expect( adapter ).not.toBeNull();

		const device = await adapter.requestDevice();
		expect( device ).toBeTruthy();

		device.destroy();

	} );

} );

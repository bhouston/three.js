/* global window, document, navigator */
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import assert from 'node:assert/strict';

const directory = process.env.HIZ_OUTPUT || '/tmp/three-ssr-hi-z';
await mkdir( directory, { recursive: true } );
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit' ]
	} );
	const page = await browser.newPage();
	const results = [];
	for ( const fixture of ( process.env.HIZ_UNITS_ONLY === '1' ? [] : [ 'hi-z', 'ssr-hi-z', 'ssr-hi-z?traversal=adaptive', 'ssr-hi-z?traversal=adaptive&minMip=1', 'ssr-hi-z?coverage=1' ] ) ) {

		const [ name, query ] = fixture.split( '?' );
		await page.goto( `http://localhost:${ server.address().port }/test/webgpu/${ name }.html${ query ? '?' + query : '' }` );
		await page.waitForFunction( () => window.hiZTestResult?.status !== 'running' && window.hiZTestResult !== undefined, { timeout: 240000 } );
		const result = await page.evaluate( () => window.hiZTestResult );
		results.push( { fixture, ...result } );
		await writeFile( `${ directory }/validation.json`, JSON.stringify( results, null, 2 ) );
		console.log( fixture, result.status, result.cases, result.message || '' );
		assert.equal( result.status, 'passed', result.message );

	}

	const unitResults = [];
	for ( const filter of ( process.env.HIZ_UNIT_FILTERS || 'SSRNode,SpecularHelpers' ).split( ',' ) ) {

		await page.goto( `http://localhost:${ server.address().port }/test/unit/UnitTestsAddons.html?filter=${ encodeURIComponent( filter ) }` );
		await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 120000 } );
		const units = await page.evaluate( () => ( {
			stats: window._QUnitStats,
			skipped: document.getElementById( 'qunit-tests' ).textContent.includes( 'SKIPPED' ),
			failures: Array.from( document.querySelectorAll( '#qunit-tests > .fail' ), node => node.textContent )
		} ) );
		const hardware = await page.evaluate( async () => {

			const { getSharedRenderer } = await import( '/test/unit/src/nodes/tsl/gpu-test-utils.js' );
			const webgpu = await getSharedRenderer( 'webgpu' );
			const info = webgpu?.backend.device?.adapterInfo || ( await navigator.gpu.requestAdapter() )?.info;
			const gpu = info && [ info.vendor, info.architecture, info.device, info.description ].join( ' ' );
			const webgl = await getSharedRenderer( 'webgl' );
			const gl = webgl?.backend.gl;
			const debug = gl?.getExtension( 'WEBGL_debug_renderer_info' );
			const rasterizer = debug && gl.getParameter( debug.UNMASKED_RENDERER_WEBGL );
			if ( ! gpu?.trim() || info.isFallbackAdapter || ! rasterizer || /software|swiftshader|llvmpipe|lavapipe/i.test( gpu + rasterizer ) ) throw new Error( 'Both unit-test backends must identify physical GPUs' );
			return { webgpu: gpu, webgl: rasterizer };

		} );
		unitResults.push( { filter, hardware, ...units } );
		await writeFile( `${ directory }/units.json`, JSON.stringify( unitResults, null, 2 ) );
		assert.equal( units.stats.failed, 0, `${ filter }: unit regressions pass` );
		assert.ok( units.stats.passed > 0, `${ filter }: tests execute` );
		assert.equal( units.skipped, false, 'GPU unit tests must execute' );
		console.log( filter, units );

	}

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

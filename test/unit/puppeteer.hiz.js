// Focused, strict hardware run: node test/unit/puppeteer.hiz.js
// Set PUPPETEER_EXECUTABLE_PATH to use an installed Chrome, or pass --headful.
// The normal addons suite can skip unavailable WebGPU. This validation rig
// requires a real adapter and fails if any Hi-Z test is skipped.

import puppeteer from 'puppeteer';
import { fileURLToPath } from 'node:url';
import { createServer } from '../../utils/server.js';

const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const server = createServer( { root } );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );

let browser;

try {

	browser = await puppeteer.launch( {
		headless: ! process.argv.includes( '--headful' ),
		args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist' ]
	} );
	const page = await browser.newPage();
	const pageErrors = [];
	page.on( 'pageerror', error => pageErrors.push( error.message ) );
	page.on( 'console', message => {

		if ( message.type() === 'warn' || message.type() === 'error' ) console.error( message.text() );
		if ( /THREE\.TSL:.*(?:Error|error)|GPUValidationError/.test( message.text() ) ) pageErrors.push( message.text() );

	} );

	await page.evaluateOnNewDocument( () => {

		window._HiZResults = { tests: [], failures: [], skips: [] };
		const timer = setInterval( () => {

			if ( window.QUnit === undefined ) return;
			clearInterval( timer );
			QUnit.log( details => {

				if ( ! details.result ) {

					window._HiZResults.failures.push( { message: details.message, actual: details.actual, expected: details.expected } );

				}

				if ( details.message && details.message.includes( 'SKIPPED:' ) ) window._HiZResults.skips.push( details.message );

			} );
			QUnit.on( 'testEnd', test => {

				window._HiZResults.tests.push( { name: test.fullName.join( ' > ' ), status: test.status } );

			} );

		}, 1 );

	} );

	const url = `http://127.0.0.1:${ server.address().port }`;
	await page.goto( url );
	const adapterInfo = await page.evaluate( async () => {

		const adapter = await navigator.gpu?.requestAdapter( { featureLevel: 'compatibility' } );
		if ( adapter === null || adapter === undefined ) return null;
		const info = adapter.info;
		return { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };

	} );

	console.log( 'WebGPU adapter:', adapterInfo );
	if ( adapterInfo === null || adapterInfo.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( adapterInfo ) ) ) {

		throw new Error( 'Hi-Z validation requires a hardware WebGPU adapter.' );

	}

	await page.goto( `${ url }/test/unit/UnitTestsAddons.html?filter=${ encodeURIComponent( '/HiZNode|SSR.*Trace|SSRValidation/' ) }` );
	await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 180000 } );
	const results = await page.evaluate( () => ( { ...window._HiZResults, stats: window._QUnitStats } ) );
	console.log( 'Hi-Z tests:', results.stats );
	for ( const test of results.tests ) {

		if ( test.status !== 'passed' ) console.error( test );

	}

	for ( const failure of results.failures ) console.error( JSON.stringify( failure ) );
	for ( const skip of results.skips ) console.error( skip );
	for ( const error of pageErrors ) console.error( error );

	if ( results.stats.total === 0 || results.tests.length !== results.stats.total || results.stats.failed > 0 || results.stats.skipped > 0 || results.skips.length > 0 || pageErrors.length > 0 ) {

		process.exitCode = 1;

	}

} finally {

	if ( browser !== undefined ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

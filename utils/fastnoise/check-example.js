// Hardware smoke test. Localhost-only hooks never enter the production example.
// node utils/fastnoise/check-example.js OUTPUT_DIRECTORY
/* global noiseTest */
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';
import { createServer } from '../server.js';
const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const output = path.resolve( process.argv[ 2 ] );
fs.mkdirSync( output, { recursive: true } );
const source = fs.readFileSync( path.join( root, 'examples/webgpu_tsl_blue_noise.html' ), 'utf8' );
const hook = `window.noiseTest = {
 get renderer() { return renderer; }, get samples() { return samples; }, get targets() { return targets; }, get atlas() { return selectedAtlas.value; }, params, noises, selectNoise,
 renderUniform(frame) { renderer.setAnimationLoop(null); params.animate = false; params.temporalBlur = false; resetAccumulation(); index.value = frame; animate(); }
}; init().catch`;
if ( ! source.includes( 'init().catch' ) ) throw new Error( 'Review test hooks.' );
const staticServer = createServer( { root } );
const handler = staticServer.listeners( 'request' )[ 0 ];
let rejectNext = false, expectedError = false;
const requests = [];
const server = http.createServer( ( req, res ) => {

	if ( req.url.includes( '/experimental/' ) ) {

		requests.push( req.url );
		if ( rejectNext ) {

			rejectNext = false; res.statusCode = 503; res.end( 'Intentional failure test' ); return;

		}

	}

	if ( new URL( req.url, 'http://localhost' ).pathname === '/examples/webgpu_tsl_blue_noise.html' ) {

		res.setHeader( 'Content-Type', 'text/html' ); res.end( source.replace( 'init().catch', hook ) );

	} else if ( req.url === '/favicon.ico' ) {

		res.statusCode = 204; res.end();

	} else handler( req, res );

} );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
let browser;
const report = { errors: [], expectedErrors: [], choices: [], requests };
try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--use-webgpu-adapter=d3d11', '--disable-software-rasterizer' ] } );
	const page = await browser.newPage();
	await page.setViewport( { width: 1100, height: 760, deviceScaleFactor: 1 } );
	const error = text => ( expectedError ? report.expectedErrors : report.errors ).push( text );
	page.on( 'pageerror', e => error( e.message ) );
	page.on( 'console', m => {

		if ( m.type() === 'error' ) error( m.text() );

	} );
	await page.goto( `http://127.0.0.1:${server.address().port}/examples/webgpu_tsl_blue_noise.html` );
	await page.waitForFunction( () => window.noiseTest?.samples >= 3, { timeout: 120000 } );
	report.backend = await page.evaluate( () => ( { webgpu: noiseTest.renderer.backend.isWebGPUBackend } ) );
	const cdp = await browser.target().createCDPSession();
	report.renderer = ( await cdp.send( 'SystemInfo.getInfo' ) ).gpu.auxAttributes.glRenderer;
	report.adapter = await page.evaluate( async () => {

		const a = await navigator.gpu.requestAdapter(); return a ? { vendor: a.info.vendor, architecture: a.info.architecture, fallback: a.info.isFallbackAdapter ?? a.isFallbackAdapter } : null;

	} );
	if ( ! /AMD|NVIDIA|Intel/i.test( report.renderer ) || /SwiftShader|llvmpipe|lavapipe|Microsoft Basic/i.test( report.renderer ) || ! report.backend.webgpu || report.adapter?.fallback !== false ) throw new Error( 'Physical WebGPU unavailable.' );
	const selects = await page.$$( '.parameters select' );
	const dropdown = selects[ 1 ];
	const names = await dropdown.$$eval( 'option', elements => elements.map( e => e.value ) );
	if ( names.length !== 8 || requests.length !== 0 ) throw new Error( 'Expected eight choices and lazy PNG loading.' );
	const annealed = 'GPU Stratified Annealed', playback = 'PR117 Light Playback';
	expectedError = true; rejectNext = true;
	await dropdown.select( annealed );
	await page.waitForFunction( () => document.getElementById( 'status' ).textContent.includes( 'Previous method restored' ) && noiseTest.samples >= 3 );
	if ( await dropdown.evaluate( e => e.value ) !== 'FAST Spatio-Temporal' ) throw new Error( 'Failed selection was not restored.' );
	expectedError = false;
	for ( const name of names ) {

		await dropdown.select( name );
		await page.waitForFunction( selected => document.getElementById( 'blueLabel' ).textContent === selected && noiseTest.samples >= 3 && document.getElementById( 'status' ).textContent === '', {}, name );
		report.choices.push( name );

	}

	const loadedRequests = requests.length;
	await dropdown.select( annealed ); await dropdown.select( playback );
	if ( requests.length !== loadedRequests ) throw new Error( 'Cached selection fetched another PNG.' );
	await page.evaluate( async () => {

		const old = noiseTest.selectNoise( 'GPU Stratified Annealed' ); noiseTest.params.noise = 'White'; await noiseTest.selectNoise( 'White' ); await old;

	} );
	if ( await page.$eval( '#blueLabel', e => e.textContent ) !== 'White' ) throw new Error( 'Stale asynchronous selection won.' );
	for ( const [ name, filename ] of [[ annealed, 'gpu-stratified-annealed' ], [ playback, 'pr117-light-seam-03' ]] ) {

		await dropdown.select( name );
		await page.waitForFunction( selected => document.getElementById( 'blueLabel' ).textContent === selected && noiseTest.samples >= 32, {}, name );
		const data = await page.evaluate( async () => {

			await noiseTest.renderer.backend.device.queue.onSubmittedWorkDone();
			return noiseTest.renderer.domElement.toDataURL().split( ',' )[ 1 ];

		} );
		fs.writeFileSync( path.join( output, filename + '-sphere.png' ), Buffer.from( data, 'base64' ) );

	}

	await selects[ 0 ].select( 'Uniform' );
	await page.$eval( '.parameters input[type=checkbox]', e => {

		e.checked = false; e.dispatchEvent( new Event( 'change', { bubbles: true } ) );

	} );
	report.uniform = [];
	for ( const [ name, filename ] of [[ annealed, 'gpu-stratified-annealed' ], [ playback, 'pr117-light-seam-03' ]] ) {

		await dropdown.select( name );
		await page.waitForFunction( selected => document.getElementById( 'blueLabel' ).textContent === selected, {}, name );
		const atlas = PNG.sync.read( fs.readFileSync( path.join( root, 'examples/textures/noises/experimental/' + filename + '.png' ) ) );
		for ( const frame of [ 0, 31, 32 ] ) {

			const state = await page.evaluate( async frame => {

				noiseTest.renderUniform( frame );
				const t = noiseTest.targets[ 1 ];
				const pixels = await noiseTest.renderer.readRenderTargetPixelsAsync( t, 0, 0, t.width, t.height );
				return { width: t.width, height: t.height, pixels: Array.from( pixels ) };

			}, frame );
			let maxError = 0;
			for ( let y = 0; y < state.height; y ++ ) for ( let x = Math.ceil( state.width / 2 ); x < state.width; x ++ ) {

				const cycle = Math.floor( frame / 32 ) % 128;
				const offset = ( ( frame % 32 * 128 + ( y + 71 * cycle ) % 128 ) * 128 + ( x + 37 * cycle ) % 128 ) * 4;
				const expected = ( atlas.data[ offset ] + .5 ) / 256;
				maxError = Math.max( maxError, Math.abs( state.pixels[ ( y * state.width + x ) * 4 ] - expected ) );

			}

			report.uniform.push( { name, frame, maxError } );
			if ( maxError > 1e-6 ) throw new Error( 'Uniform atlas sampling mismatch: ' + JSON.stringify( report.uniform.at( - 1 ) ) );

		}

		const data = await page.evaluate( () => noiseTest.renderer.domElement.toDataURL().split( ',' )[ 1 ] );
		fs.writeFileSync( path.join( output, filename + '-canvas.png' ), Buffer.from( data, 'base64' ) );

	}

	report.inspector = await page.evaluate( () => noiseTest.renderer.inspector.console.logContainer.textContent );
	if ( report.errors.length ) throw new Error( report.errors.join( '\n' ) );
	report.pass = true;
	console.log( 'PASS: 8 dropdown choices, load failure/retry, lazy cache, race guard, both atlases at frames 0/31/32; physical GPU.' );

} finally {

	fs.writeFileSync( path.join( output, 'example-check.json' ), JSON.stringify( report, null, 2 ) );
	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

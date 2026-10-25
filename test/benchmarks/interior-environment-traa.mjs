/* global window, document, requestAnimationFrame */
// Checks hallway temporal stability with only environment lighting and optional GTAO/GTSO.
// TRAA_REDUCED_AO_GUIDES=1 restores the regression as a negative control.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import process from 'node:process';
import console from 'node:console';

const directory = process.env.TRAA_OUTPUT || '/tmp/three-interior-environment-traa';
await mkdir( directory, { recursive: true } );
let html = ( await readFile( 'examples/webgpu_interior.html', 'utf8' ) )
	.replace( "model: 'Dual Rooms'", "model: 'Space Ship Hallway'" )
	.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
	.replace( 'renderer.setAnimationLoop( animate );', `window.__traaTest = { renderer, camera, controls, params, ssrParams, ssgiParams, prePass, scenePass, traaPass, renderPipeline, updateAO, updateDirectLights, updateIndirectLighting, animate }; renderer.setAnimationLoop( animate );` );
if ( process.env.TRAA_REDUCED_AO_GUIDES === '1' ) html = html.replace( 'ao( prePassDepth, prePassNormal, camera )', 'ao( effectDepth, effectNormal, camera )' );
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer' ],
		defaultViewport: { width: 960, height: 720, deviceScaleFactor: 2 }
	} );
	const page = await browser.newPage();
	await page.setViewport( { width: 960, height: 720, deviceScaleFactor: 2 } );
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => { if ( message.type() === 'error' ) errors.push( message.text() ); } );
	await page.setRequestInterception( true );
	page.on( 'request', request => request.url().endsWith( '/examples/webgpu_interior.html' )
		? request.respond( { status: 200, contentType: 'text/html', body: html } )
		: request.url().endsWith( '/favicon.ico' ) ? request.respond( { status: 204 } ) : request.continue() );
	await page.goto( `http://localhost:${ server.address().port }/examples/webgpu_interior.html` );
	await page.waitForFunction( () => window.__traaTest && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 240000 } );
	const hardware = await page.evaluate( () => {

		const info = window.__traaTest.renderer.backend.device.adapterInfo;
		if ( ! info.vendor || info.isFallbackAdapter || /software|swiftshader|llvmpipe|lavapipe/i.test( info.vendor + info.description ) ) throw new Error( 'Physical WebGPU hardware is required' );
		return { vendor: info.vendor, architecture: info.architecture, description: info.description };

	} );
	await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
	await page.evaluate( async () => {

		const s = window.__traaTest;
		s.renderer.setAnimationLoop( null );
		s.renderer._animation.stop();
		s.controls.enableDamping = false;
		s.controls.update();
		s.ssrParams.enabled = false;
		s.ssgiParams.enabled = false;
		Object.assign( s.params, {
			diffuseBaking: false, reflectionSpecular: false, reflectionDiffuse: false,
			lightProbeDiffuse: false, directLights: false, output: 'Beauty'
		} );
		s.updateDirectLights();
		s.updateIndirectLighting();
		// Let Inspector listen() synchronize the programmatic checkbox changes.
		await new Promise( resolve => requestAnimationFrame( () => requestAnimationFrame( resolve ) ) );
		window.__stepTRAA = async count => {

			for ( let i = 0; i < count; i ++ ) {

				s.renderer._nodes.nodeFrame.update();
				s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId;
				s.animate();

			}

			await s.renderer.backend.device.queue.onSubmittedWorkDone();

		};

	} );
	const results = [];
	for ( const ao of [ 'GTAO + GTSO', 'None' ] ) {

		const result = await page.evaluate( async ao => {

			const s = window.__traaTest;
			s.params.aoType = ao;
			s.updateAO();
			await window.__stepTRAA( 96 );
			const { DataUtils } = await import( '/src/Three.WebGPU.js' );
			const read = async target => {

				const data = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height );
				return Float32Array.from( data, value => data instanceof Uint16Array ? DataUtils.fromHalfFloat( value ) : value );

			};
			let previousRaw, previousAA;
			let rawDifference = 0, aaDifference = 0, samples = 0;
			const firstJitter = s.traaPass._jitterIndex;
			for ( let i = 0; i < 24; i ++ ) {

				await window.__stepTRAA( 1 );
				const raw = await read( s.scenePass.renderTarget );
				const resolved = s.traaPass._resolveRenderTarget;
				if ( resolved.width !== s.prePass.getTexture( 'depth' ).image.width || resolved.height !== s.prePass.getTexture( 'depth' ).image.height ) throw new Error( 'TRAA lost its full-resolution history' );
				const aa = await read( resolved );
				if ( raw.length !== aa.length || ! aa.every( Number.isFinite ) ) throw new Error( 'TRAA output is invalid' );
				if ( previousRaw ) {

					for ( let j = 0; j < raw.length; j ++ ) {

						if ( j % 4 === 3 ) continue;
						rawDifference += ( raw[ j ] - previousRaw[ j ] ) ** 2;
						aaDifference += ( aa[ j ] - previousAA[ j ] ) ** 2;
						samples ++;

					}

				}

				previousRaw = raw;
				previousAA = aa;

			}

			if ( s.traaPass._jitterIndex !== ( firstJitter + 24 ) % 32 ) throw new Error( 'TRAA must advance once per frame' );
			return { ao, rawRMSE: Math.sqrt( rawDifference / samples ), resolvedRMSE: Math.sqrt( aaDifference / samples ) };

		}, ao );
		results.push( result );
		console.log( result );
		await page.screenshot( { path: `${ directory }/${ ao === 'None' ? 'environment-only' : 'environment-ao' }.png` } );

	}
	await writeFile( `${ directory }/validation.json`, JSON.stringify( { hardware, results, errors }, null, 2 ) );
	assert.deepEqual( errors, [], 'No browser or GPU errors' );
	for ( const result of results ) assert.ok( result.resolvedRMSE < result.rawRMSE * 0.15, 'TRAA substantially reduces temporal variation' );
	assert.ok( results[ 0 ].resolvedRMSE < results[ 1 ].resolvedRMSE * 1.5, 'GTAO/GTSO does not introduce excessive residual temporal noise' );
	console.log( { hardware, status: 'passed' } );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

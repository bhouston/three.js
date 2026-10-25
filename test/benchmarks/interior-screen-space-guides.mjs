/* global window, document */
// Hardware integration checks for the interior example's shared screen-space grid.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import process from 'node:process';
import console from 'node:console';

const directory = process.env.GUIDES_OUTPUT || '/tmp/three-interior-screen-space-guides';
await mkdir( directory, { recursive: true } );
const html = ( await readFile( 'examples/webgpu_interior.html', 'utf8' ) )
	.replace( "model: 'Dual Rooms'", "model: 'Primitives Floor (Color Dielectric)'" )
	.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
	.replace( 'renderer.setAnimationLoop( animate );', `window.__guidesTest = { renderer, camera, controls, params, prePass, screenSpaceGuides, ssrPass, ssgiPass, aoPass, ssrParams, updateSSRParameters, updateScreenSpaceResolution, updateOutput, animate }; renderer.setAnimationLoop( animate );` );
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer' ],
		defaultViewport: { width: 1000, height: 750, deviceScaleFactor: 2 }
	} );
	const page = await browser.newPage();
	await page.setViewport( { width: 1000, height: 750, deviceScaleFactor: 2 } );
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => { if ( message.type() === 'error' ) errors.push( message.text() ); } );
	await page.setRequestInterception( true );
	page.on( 'request', request => request.url().endsWith( '/examples/webgpu_interior.html' )
		? request.respond( { status: 200, contentType: 'text/html', body: html } )
		: request.url().endsWith( '/favicon.ico' ) ? request.respond( { status: 204 } ) : request.continue() );
	await page.goto( `http://localhost:${ server.address().port }/examples/webgpu_interior.html` );
	await page.waitForFunction( () => window.__guidesTest && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 240000 } );
	const hardware = await page.evaluate( () => {

		const info = window.__guidesTest.renderer.backend.device.adapterInfo;
		if ( ! info.vendor || info.isFallbackAdapter || /software|swiftshader|llvmpipe|lavapipe/i.test( info.vendor + info.description ) ) throw new Error( 'Physical WebGPU hardware is required' );
		return { vendor: info.vendor, architecture: info.architecture, description: info.description };

	} );
	await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
	await page.evaluate( () => {

		const s = window.__guidesTest;
		s.renderer.setAnimationLoop( null );
		s.renderer._animation.stop();
		s.controls.enableDamping = false;
		window.__stepGuides = async count => {

			for ( let i = 0; i < count; i ++ ) {

				s.renderer._nodes.nodeFrame.update();
				s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId;
				s.animate();

			}

			await s.renderer.backend.device.queue.onSubmittedWorkDone();

		};

	} );
	const results = [];
	for ( const scale of [ 0.5, 1, 0.25, 0.5 ] ) {

		await page.evaluate( scale => {

			const s = window.__guidesTest;
			s.params.resolutionScale = scale;
			s.updateScreenSpaceResolution();

		}, scale );
		for ( const output of [ 'Beauty', 'SSR Coverage', 'SSGI Coverage', 'AO' ] ) {

			const result = await page.evaluate( async output => {

				const s = window.__guidesTest;
				s.params.output = output;
				s.updateOutput();
				await window.__stepGuides( 24 );
				const image = s.prePass.getTexture( 'depth' ).image;
				const guides = s.screenSpaceGuides.getRenderTarget();
				const ssr = s.ssrPass.getRenderTarget();
				const ssgi = s.ssgiPass.getGINode().value.image;
				const ao = s.aoPass.getTextureNode().value.image;
				const data = await s.renderer.readRenderTargetPixelsAsync( guides, 0, 0, guides.width, guides.height );
				if ( ! data.every( Number.isFinite ) ) throw new Error( 'Prepared guides contain nonfinite values' );
				return {
					output, source: [ image.width, image.height ],
					guides: [ guides.width, guides.height ], ssr: [ ssr.width, ssr.height ],
					ssgi: [ ssgi.width, ssgi.height ], ao: [ ao.width, ao.height ]
				};

			}, output );
			const expected = result.source.map( size => Math.round( size * scale ) );
			for ( const key of [ 'guides', 'ssr', 'ssgi', 'ao' ] ) assert.deepEqual( result[ key ], expected, `${ key } uses the shared grid` );
			results.push( { scale, ...result } );
			await page.screenshot( { path: `${ directory }/${ scale }-${ output.replaceAll( ' ', '-' ) }.png` } );

		}

	}
	// Both hierarchy traversal paths must consume the same prepared depth.
	for ( const traversal of [ 'Conservative Hi-Z', 'Adaptive Hi-Z (experimental)' ] ) {

		await page.evaluate( async traversal => {

			const s = window.__guidesTest;
			s.ssrParams.traversal = traversal;
			s.updateSSRParameters();
			s.params.output = 'SSR Coverage';
			s.updateOutput();
			await window.__stepGuides( 8 );

		}, traversal );

	}
	// Resize while retaining the same pass graph and history resources.
	await page.setViewport( { width: 963, height: 723, deviceScaleFactor: 2 } );
	await page.evaluate( async () => {

		await window.__stepGuides( 8 );
		const s = window.__guidesTest;
		const image = s.prePass.getTexture( 'depth' ).image;
		const guides = s.screenSpaceGuides.getRenderTarget();
		if ( guides.width !== Math.round( image.width * s.params.resolutionScale ) || guides.height !== Math.round( image.height * s.params.resolutionScale ) ) throw new Error( 'Guide size did not follow viewport resize' );

	} );
	assert.deepEqual( errors, [], 'No browser or GPU validation errors' );
	await writeFile( `${ directory }/validation.json`, JSON.stringify( { hardware, results, errors }, null, 2 ) );
	console.log( { hardware, cases: results.length, errors } );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

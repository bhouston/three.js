/* global window, document, navigator */
// Hardware regression for Interior's SSR fallback. Run from the repository root.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import assert from 'node:assert/strict';

const directory = process.env.SSR_FALLBACK_OUTPUT || '/tmp/three-interior-ssr-fallback';
await mkdir( directory, { recursive: true } );
const html = ( await readFile( process.env.SSR_FALLBACK_SOURCE || 'examples/webgpu_interior.html', 'utf8' ) )
	.replace( "model: 'Dual Rooms'", "model: 'Damaged Helmet'" )
	.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
	.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
	.replace( 'renderer.setAnimationLoop( animate );', `window.__fallbackTest = { renderer, scene, camera, params, prePass, scenePass, ssrPass, ssrParams, controls, animate, updateIndirectLighting, updateSSRParameters }; renderer.setAnimationLoop( animate );` );
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer' ],
		defaultViewport: { width: 960, height: 720 }
	} );
	const page = await browser.newPage();
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => { if ( message.type() === 'error' ) errors.push( message.text() ); } );
	await page.setRequestInterception( true );
	page.on( 'request', request => request.url().endsWith( '/favicon.ico' )
		? request.respond( { status: 204 } )
		: request.url().endsWith( '/examples/webgpu_interior.html' )
			? request.respond( { status: 200, contentType: 'text/html', body: html } )
			: request.continue() );
	const origin = `http://localhost:${ server.address().port }`;
	await page.goto( origin );
	const hardware = await page.evaluate( async () => {

		const info = ( await navigator.gpu.requestAdapter() )?.info;
		if ( ! info || ! info.vendor || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( info ) ) ) throw new Error( 'A hardware WebGPU adapter is required' );
		return { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter };

	} );
	await page.goto( `${ origin }/examples/webgpu_interior.html` );
	await page.waitForFunction( () => window.__fallbackTest && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 240000 } );
	await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
	const results = await page.evaluate( async () => {

		const s = window.__fallbackTest;
		s.renderer.setAnimationLoop( null );
		s.renderer._animation.stop();
		s.controls.enableDamping = false;
		const { DataUtils } = await import( '/src/Three.WebGPU.js' );
		const drain = () => s.renderer.backend.device.queue.onSubmittedWorkDone();
		const step = () => {

			s.renderer.info.reset();
			s.renderer._nodes.nodeFrame.update();
			s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId;
			s.animate();

		};
		window.__fallbackCapture = async mode => {

			await drain();
			s.ssrParams.enabled = mode !== 'disabled';
			s.ssrParams.denoise = mode !== 'miss-unfiltered';
			s.updateIndirectLighting();
			s.updateSSRParameters();
			// Zero march steps guarantee no screen hits without changing fallback.
			if ( mode.startsWith( 'miss-' ) ) s.ssrPass.quality.value = 0;
			for ( let i = 0; i < 64; i ++ ) step();
			await drain();
			// Fixed visor crop avoids background dilution. Compare linear HDR beauty,
			// before tone mapping/TRAA; the visor has negligible screen-hit coverage.
			const values = await s.renderer.readRenderTargetPixelsAsync( s.scenePass.renderTarget, 420, 260, 75, 50 );
			return Array.from( values, DataUtils.fromHalfFloat );

		};

		const distance = s.camera.position.distanceTo( s.controls.target );
		window.__fallbackHDRCapture = async mode => {

			s.camera.position.set( distance * 0.94 * Math.SQRT1_2, distance * 0.34, distance * 0.94 * Math.SQRT1_2 ).add( s.controls.target );
			s.controls.update();
			s.params.reflectionSpecular = false;
			s.scene.background = null;
			await window.__fallbackCapture( mode );
			const target = s.scenePass.renderTarget;
			const data = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height );
			const luminance = [];
			for ( let i = 0; i < data.length; i += 4 ) {

				luminance.push( DataUtils.fromHalfFloat( data[ i ] ) * 0.2126 + DataUtils.fromHalfFloat( data[ i + 1 ] ) * 0.7152 + DataUtils.fromHalfFloat( data[ i + 2 ] ) * 0.0722 );

			}
			// Average the brightest pixels to tolerate subpixel highlight movement.
			luminance.sort( ( a, b ) => b - a );
			return { peak: luminance[ 0 ], top100: luminance.slice( 0, 100 ).reduce( ( sum, value ) => sum + value, 0 ) / 100 };

		};
		return {};

	} );
	const images = {};
	for ( const mode of [ 'disabled', 'default', 'miss-unfiltered', 'miss-filtered' ] ) {

		images[ mode ] = await page.evaluate( mode => window.__fallbackCapture( mode ), mode );
		await page.screenshot( { path: `${ directory }/${ mode }.png` } );

	}
	for ( const mode of [ 'default', 'miss-unfiltered', 'miss-filtered' ] ) {

		let error = 0, energy = 0;
		for ( let i = 0; i < images.disabled.length; i ++ ) {

			if ( i % 4 === 3 ) continue;
			assert.ok( Number.isFinite( images[ mode ][ i ] ), `${ mode }: finite radiance` );
			error += ( images[ mode ][ i ] - images.disabled[ i ] ) ** 2;
			energy += images.disabled[ i ] ** 2;

		}
		results[ mode ] = { relativeRMSE: Math.sqrt( error / Math.max( energy, 1e-8 ) ) };

	}
	const highlights = {};
	for ( const mode of [ 'disabled', 'default', 'miss-unfiltered', 'miss-filtered' ] ) {

		highlights[ mode ] = await page.evaluate( mode => window.__fallbackHDRCapture( mode ), mode );
		await page.screenshot( { path: `${ directory }/hdr-${ mode }.png` } );

	}
	const report = { hardware, results, highlights, errors };
	await writeFile( `${ directory }/results.json`, JSON.stringify( report, null, 2 ) );
	console.log( report );
	assert.deepEqual( errors, [], 'No browser/GPU errors' );
	// The unfiltered miss case isolates fallback correctness. Filtering introduces
	// additional spatial/temporal approximation, so use a wider bound there.
	for ( const [ mode, result ] of Object.entries( results ) ) {

		const tolerance = mode === 'miss-unfiltered' ? 0.1 : 0.25;
		assert.ok( result.relativeRMSE < tolerance, `${ mode }: fallback changed visor radiance (${ result.relativeRMSE })` );

	}

	assert.ok( highlights.disabled.top100 > 10, 'The reference contains bright HDR sun reflections' );
	for ( const mode of [ 'default', 'miss-unfiltered', 'miss-filtered' ] ) {

		assert.ok( highlights[ mode ].top100 > highlights.disabled.top100 * 0.5, `${ mode }: preserve HDR highlight energy (${ highlights[ mode ].top100 } vs ${ highlights.disabled.top100 })` );

	}

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

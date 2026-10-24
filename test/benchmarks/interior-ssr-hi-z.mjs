/* global window, document, navigator, performance */
// Compare Interior SSR traversal quality and GPU-completed full-frame throughput. Run from the repository root.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import assert from 'node:assert/strict';

const directory = process.env.HIZ_OUTPUT || '/tmp/three-interior-ssr-hi-z';
const model = process.env.HIZ_MODEL || 'Primitives Floor (Metallic Mirror)';
await mkdir( directory, { recursive: true } );
const html = ( await readFile( 'examples/webgpu_interior.html', 'utf8' ) )
	.replace( "model: 'Dual Rooms'", `model: ${ JSON.stringify( model ) }` )
	.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
	.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
	.replace( "renderer.copyTextureToTexture( scenePass.getTexture( 'output' ), lightingHistory.texture );", "if ( ! window.__freezeLighting ) renderer.copyTextureToTexture( scenePass.getTexture( 'output' ), lightingHistory.texture );" )
	.replace( 'renderer.setAnimationLoop( animate );', `window.__fallbackTest = { renderer, scene, camera, params, prePass, scenePass, lightingHistory, ssrPass, ssrParams, controls, animate, updateOutput, updateIndirectLighting, updateSSRParameters }; renderer.setAnimationLoop( animate );` );
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit' ],
		defaultViewport: { width: Number( process.env.HIZ_WIDTH || 960 ), height: Number( process.env.HIZ_HEIGHT || 720 ) }
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
	const setup = await page.evaluate( async () => {

		const s = window.__fallbackTest;
		s.renderer.setAnimationLoop( null );
		s.renderer._animation.stop();
		s.controls.enableDamping = false;
		const queue = s.renderer.backend.device.queue;
		window.__drain = () => queue.onSubmittedWorkDone();
		window.__step = () => {

			s.renderer.info.reset();
			s.renderer._nodes.nodeFrame.update();
			s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId;
			s.animate();

		};
		for ( let i = 0; i < 64; i ++ ) window.__step();
		await window.__drain();
		window.__freezeLighting = true;
		window.__frozenLighting = s.lightingHistory.clone();
		s.renderer.initRenderTarget( window.__frozenLighting );
		s.renderer.copyTextureToTexture( s.lightingHistory.texture, window.__frozenLighting.texture );
		window.__pyramidBuilds = 0;
		const pyramid = s.ssrPass.hiZ;
		const update = pyramid.updateBefore.bind( pyramid );
		pyramid.updateBefore = frame => { window.__pyramidBuilds ++; update( frame ); };
		await window.__drain();
		return { width: s.renderer.domElement.width, height: s.renderer.domElement.height };

	} );
	const captures = {};
	const results = [];
	for ( const mode of [ 'Linear', 'Conservative Hi-Z', 'Adaptive Hi-Z (experimental)', 'Linear' ] ) {

		const capture = await page.evaluate( async mode => {

			const s = window.__fallbackTest;
			await window.__drain();
			s.params.output = 'SSR Raw';
			s.updateOutput();
			s.ssrParams.traversal = mode;
			s.updateSSRParameters();
			for ( let i = 0; i < 16; i ++ ) window.__step();
			await window.__drain();
			// Fixed stochastic sequence and frozen source isolate traversal.
			s.ssrPass._noiseIndex.value = 0;
			s.renderer._nodes.nodeFrame.frameId = Math.ceil( s.renderer._nodes.nodeFrame.frameId / 96 ) * 96;
			window.__step();
			await window.__drain();
			const target = s.ssrPass.getRenderTarget();
			const rgba = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height );
			const coverage = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, 1 );
			const { DataUtils } = await import( '/src/Three.WebGPU.js' );
			return { rgba: Array.from( rgba, DataUtils.fromHalfFloat ), coverage: Array.from( coverage, DataUtils.fromHalfFloat ) };

		}, mode );
		let maxError = 0, rgbError = 0, changedCoverage = 0;
		if ( captures.Linear ) {

			for ( let i = 0; i < capture.rgba.length; i ++ ) {

				assert.ok( Number.isFinite( capture.rgba[ i ] ), 'Finite trace output' );
				maxError = Math.max( maxError, Math.abs( capture.rgba[ i ] - captures.Linear.rgba[ i ] ) );
				if ( i % 4 !== 3 ) rgbError += ( capture.rgba[ i ] - captures.Linear.rgba[ i ] ) ** 2;
				if ( i % 4 === 0 && capture.coverage[ i ] !== captures.Linear.coverage[ i ] ) changedCoverage ++;

			}

		}
		captures[ mode ] = capture;
		await page.screenshot( { path: `${ directory }/${ results.length }.png` } );
		const throughput = await page.evaluate( async () => {

			const s = window.__fallbackTest;
			s.params.output = 'Beauty';
			s.updateOutput();
			window.__freezeLighting = false;
			const run = async duration => {

				let frames = 0;
				const buildsBefore = window.__pyramidBuilds;
				const before = s.ssrPass._noiseIndex.value;
				const start = performance.now();
				do {

					for ( let i = 0; i < 4; i ++ ) {

						window.__step();

					}
					await window.__drain();
					frames += 4;

				} while ( performance.now() - start < duration );
				const seconds = ( performance.now() - start ) / 1000;
				if ( s.ssrPass._noiseIndex.value - before !== frames ) throw new Error( 'SSR must execute once per completed frame' );
				const pyramidBuilds = window.__pyramidBuilds - buildsBefore;
				if ( pyramidBuilds !== ( s.ssrPass.hiZEnabled ? frames : 0 ) ) throw new Error( `Unexpected pyramid builds: ${ pyramidBuilds } for ${ frames } frames` );
				return { frames, seconds, fps: frames / seconds, pyramidBuilds };

			};
			await run( 1000 );
			await window.__drain();
			const result = await run( 5000 );
			await window.__drain();
			window.__freezeLighting = true;
			s.renderer.copyTextureToTexture( window.__frozenLighting.texture, s.lightingHistory.texture );
			await window.__drain();
			return result;

		} );
		results.push( { mode, maxError, rgbRMSE: Math.sqrt( rgbError / ( capture.rgba.length * 0.75 ) ), changedCoverage, ...throughput } );
		console.log( results.at( - 1 ) );
		await writeFile( `${ directory }/results.json`, JSON.stringify( { hardware, setup, model, results, errors }, null, 2 ) );

	}
	assert.deepEqual( errors, [], 'No browser/GPU errors' );
	for ( const result of results.filter( row => row.mode === 'Conservative Hi-Z' || row === results.at( - 1 ) ) ) {

		assert.equal( result.maxError, 0, 'Conservative and repeated linear traces match the baseline exactly' );
		assert.equal( result.changedCoverage, 0, 'Coverage matches exactly' );

	}

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

/* global window, document, navigator, performance */
// Compare Interior SSR traversal quality and GPU-completed full-frame throughput. Run from the repository root.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import assert from 'node:assert/strict';

const directory = process.env.HIZ_OUTPUT || '/tmp/three-interior-ssr-hi-z';
const hierarchySource = process.env.HIZ_NODE_SOURCE ? await readFile( process.env.HIZ_NODE_SOURCE, 'utf8' ) : null;
const diagnostic = process.env.HIZ_DIAGNOSTIC === '1';
let reflectionSource = process.env.HIZ_SSR_SOURCE ? await readFile( process.env.HIZ_SSR_SOURCE, 'utf8' ) : null;

if ( diagnostic ) {

	// Compile counters only for this untimed diagnostic; production shaders and
	// throughput windows contain no instrumentation.
	reflectionSource ??= await readFile( 'examples/jsm/tsl/display/SSRNode.js', 'utf8' );
	reflectionSource = reflectionSource
		.replace( 'const hitD = float( 0 ).toVar();', 'const hitD = float( 0 ).toVar(); const traceIterations = float( 0 ).toVar(); const boundChecks = float( 0 ).toVar(); const skippedSamples = float( 0 ).toVar();' )
		.replace( 'const s = adaptive ?', 'traceIterations.addAssign( 1 ); const s = adaptive ?' )
		.replace( 'If( canSkipBatch(', 'boundChecks.addAssign( 1 ); If( canSkipBatch(' )
		.replace( 'i.assign( last );', 'skippedSamples.addAssign( float( last.sub( i ).add( 1 ) ) ); i.assign( last );' )
		.replace( 'return output.max( 0 );', 'return vec4( traceIterations, boundChecks, skippedSamples, foundHit.select( 1, traceIterations.greaterThanEqual( iterationEnd.sub( 1 ) ).select( 2, 0 ) ) );' );

}
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
	page.on( 'request', request => {

		if ( hierarchySource && request.url().endsWith( '/HiZNode.js' ) ) return request.respond( { status: 200, contentType: 'application/javascript', body: hierarchySource } );
		if ( reflectionSource && request.url().endsWith( '/SSRNode.js' ) ) return request.respond( { status: 200, contentType: 'application/javascript', body: reflectionSource } );
		if ( request.url().endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
		if ( request.url().endsWith( '/examples/webgpu_interior.html' ) ) return request.respond( { status: 200, contentType: 'text/html', body: html } );
		return request.continue();

	} );
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
	await page.evaluate( value => { window.__hizDiagnostic = value; }, diagnostic );
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
			let rejectedByRoughness = 0, rejectedIterations = 0;
			if ( window.__hizDiagnostic ) {

				const guideIndex = s.prePass.renderTarget.textures.indexOf( s.prePass.getTexture( 'output' ) );
				const guides = await s.renderer.readRenderTargetPixelsAsync( s.prePass.renderTarget, 0, 0, target.width, target.height, guideIndex );
				const decode = guides instanceof Uint16Array ? DataUtils.fromHalfFloat : guides instanceof Uint8Array ? value => value / 255 : value => value;
				for ( let i = 0; i < rgba.length; i += 4 ) {

					const count = DataUtils.fromHalfFloat( rgba[ i ] );
					if ( count > 0 && decode( guides[ i + 3 ] ) >= s.ssrParams.maxRoughness ) {

						rejectedByRoughness ++;
						rejectedIterations += count;

					}

				}

			}
			return { rgba: Array.from( rgba, DataUtils.fromHalfFloat ), coverage: Array.from( coverage, DataUtils.fromHalfFloat ), rejectedByRoughness, rejectedIterations };

		}, mode );
		if ( diagnostic ) {

			let active = 0, iterations = 0, checks = 0, skipped = 0, hits = 0, exhausted = 0;
			const histogram = {};
			for ( let i = 0; i < capture.rgba.length; i += 4 ) {

				const count = capture.rgba[ i ];
				if ( count === 0 ) continue;
				active ++;
				iterations += count;
				checks += capture.rgba[ i + 1 ];
				skipped += capture.rgba[ i + 2 ];
				hits += capture.rgba[ i + 3 ] === 1 ? 1 : 0;
				exhausted += capture.rgba[ i + 3 ] === 2 ? 1 : 0;
				histogram[ count ] = ( histogram[ count ] || 0 ) + 1;

			}
			results.push( { mode, active, rejectedByRoughness: capture.rejectedByRoughness, rejectedIterations: capture.rejectedIterations, meanIterations: iterations / active, checks, skipped, hits, exhausted, histogram } );
			await writeFile( `${ directory }/diagnostic.json`, JSON.stringify( { hardware, setup, model, results, errors }, null, 2 ) );
			console.log( { ...results.at( - 1 ), histogram: undefined } );
			continue;

		}
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
		await writeFile( `${ directory }/results.json`, JSON.stringify( { hardware, browser: await browser.version(), hierarchySource: process.env.HIZ_NODE_SOURCE || null, reflectionSource: process.env.HIZ_SSR_SOURCE || null, setup, model, results, errors }, null, 2 ) );

	}
	assert.deepEqual( errors, [], 'No browser/GPU errors' );
	for ( const result of ( diagnostic ? [] : results ).filter( row => row.mode === 'Conservative Hi-Z' || row === results.at( - 1 ) ) ) {

		assert.equal( result.maxError, 0, 'Conservative and repeated linear traces match the baseline exactly' );
		assert.equal( result.changedCoverage, 0, 'Coverage matches exactly' );

	}

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

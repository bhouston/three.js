/* global window, document, navigator, performance */
// Compare early classification with tracing all rays, using the same fallback contract. Run from the repository root.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import assert from 'node:assert/strict';

const directory = process.env.CLASSIFY_OUTPUT || '/tmp/three-ssr-classification';
const model = process.env.CLASSIFY_MODEL || 'Primitives Floor (Metallic Mirror)';
await mkdir( directory, { recursive: true } );
const html = ( await readFile( 'examples/webgpu_interior.html', 'utf8' ) )
	.replace( "model: 'Dual Rooms'", `model: ${ JSON.stringify( model ) }` )
	.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
	.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
	.replace( "renderer.copyTextureToTexture( scenePass.getTexture( 'output' ), lightingHistory.texture );", "if ( ! window.__freezeLighting ) renderer.copyTextureToTexture( scenePass.getTexture( 'output' ), lightingHistory.texture );" )
	.replace( 'const resolvedTexture = convertToTexture( indirectSpecularRaw.sample( uv() ) );', 'const resolvedTexture = convertToTexture( indirectSpecularRaw.sample( uv() ) ); window.__classificationResolved = resolvedTexture;' )
	.replace( 'renderer.setAnimationLoop( animate );', `window.__fallbackTest = { renderer, scene, camera, params, prePass, scenePass, lightingHistory, ssrPass, ssrParams, ssrTemporal, ssrDenoise, ssrCoverageDenoise, controls, animate, updateOutput, updateIndirectLighting, updateSSRParameters }; renderer.setAnimationLoop( animate );` );
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit' ],
		defaultViewport: { width: Number( process.env.CLASSIFY_WIDTH || 960 ), height: Number( process.env.CLASSIFY_HEIGHT || 720 ) }
	} );
	const page = await browser.newPage();
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => { if ( message.type() === 'error' ) errors.push( message.text() ); } );
	await page.setRequestInterception( true );
	page.on( 'request', request => request.url().endsWith( '/examples/webgpu_interior.html' )
		? request.respond( { status: 200, contentType: 'text/html', body: html } )
		: request.url().endsWith( '/favicon.ico' ) ? request.respond( { status: 204 } ) : request.continue() );
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
	await page.evaluate( async () => {

		const s = window.__fallbackTest;
		const info = s.renderer.backend.device.adapterInfo;
		if ( info && ( info.isFallbackAdapter || /software|swiftshader|llvmpipe|lavapipe/i.test( [ info.vendor, info.description ].join( ' ' ) ) ) ) throw new Error( 'Software rendering rejected' );
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
		s.params.output = 'Indirect Specular Filtered';
		s.updateOutput();
		for ( let i = 0; i < 64; i ++ ) window.__step();
		await window.__drain();
		window.__freezeLighting = true;
		window.__frozenLighting = s.lightingHistory.clone();
		s.renderer.initRenderTarget( window.__frozenLighting );
		s.renderer.copyTextureToTexture( s.lightingHistory.texture, window.__frozenLighting.texture );
		window.__cameraStart = s.camera.position.clone();
		window.__cameraDistance = s.camera.position.distanceTo( s.controls.target );
		await window.__drain();

	} );
	const quality = [];
	if ( process.env.CLASSIFY_SKIP_QUALITY !== '1' ) {

		for ( const traversal of [ 'Linear', 'Conservative Hi-Z', 'Adaptive Hi-Z (experimental)' ] ) {

			for ( const resolveOrder of [ 'Before filtering', 'After filtering (coverage-aware)' ] ) {

				for ( const resolutionScale of [ 1, 0.5 ] ) {

					const comparisons = await page.evaluate( async settings => {

						const s = window.__fallbackTest;
						const { DataUtils } = await import( '/src/Three.WebGPU.js' );
						const read = async target => {

							const data = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height );
							return Float32Array.from( data, DataUtils.fromHalfFloat );

						};
						const capture = async classify => {

							await window.__drain();
							Object.assign( s.ssrParams, settings, { classify } );
							s.camera.position.copy( window.__cameraStart );
							s.controls.update();
							s.updateIndirectLighting();
							s.updateSSRParameters();
							s.params.output = 'Indirect Specular Filtered';
							s.updateOutput();
							s.ssrPass._noiseIndex.value = 0;
							// Replay exactly the same stochastic sequence and camera path.
							s.renderer._nodes.nodeFrame.frameId = 10000;
							const samples = [];
							for ( let i = 0; i < 32; i ++ ) {

								const move = i < 16 ? 0 : ( i - 15 ) * window.__cameraDistance * 0.0005;
								s.camera.position.copy( window.__cameraStart ).addScaledVector( s.camera.up, move );
								window.__step();
								if ( i === 15 || i === 31 ) {

									await window.__drain();
									// This RTT is not a dependency of the coverage-aware output.
									// Render it explicitly so its comparison cannot read stale data.
									window.__classificationResolved.updateBefore( { renderer: s.renderer } );
									await window.__drain();
									samples.push( { resolved: await read( window.__classificationResolved.renderTarget ), filtered: await read( settings.resolveOrder === 'Before filtering' ? s.ssrDenoise.getRenderTarget() : s.ssrCoverageDenoise.getRenderTarget() ) } );

								}

							}
							if ( s.ssrPass._noiseIndex.value !== 32 ) throw new Error( 'Quality captures must execute every SSR frame' );
							return samples;

						};
						const before = await capture( false );
						const classified = await capture( true );
						const repeated = await capture( false );
						const compare = ( a, b ) => {

							let maxError = 0, squaredError = 0, rgbEnergy = 0;
							for ( let i = 0; i < a.length; i ++ ) {

								if ( ! Number.isFinite( a[ i ] ) || ! Number.isFinite( b[ i ] ) ) throw new Error( 'Nonfinite reconstruction' );
								const difference = Math.abs( a[ i ] - b[ i ] );
								maxError = Math.max( maxError, difference );
								squaredError += difference ** 2;
								if ( i % 4 !== 3 ) rgbEnergy += a[ i ] ** 2;

							}
							if ( rgbEnergy === 0 ) throw new Error( 'Quality comparison requires nonzero lighting' );
							return { maxError, rmse: Math.sqrt( squaredError / a.length ) };

						};
						return before.map( ( sample, i ) => ( {
							motion: i === 1,
							resolved: compare( sample.resolved, classified[ i ].resolved ),
							filtered: compare( sample.filtered, classified[ i ].filtered ),
							repeatedResolved: compare( sample.resolved, repeated[ i ].resolved ),
							repeatedFiltered: compare( sample.filtered, repeated[ i ].filtered )
						} ) );

					}, { traversal, resolveOrder, resolutionScale } );
					quality.push( { traversal, resolveOrder, resolutionScale, comparisons } );
					await writeFile( `${ directory }/quality.json`, JSON.stringify( { hardware, model, quality, errors }, null, 2 ) );
					console.log( 'Quality', traversal, resolveOrder, resolutionScale, 'stationary/moving max errors:', comparisons.map( row => Math.max( row.resolved.maxError, row.filtered.maxError, row.repeatedResolved.maxError, row.repeatedFiltered.maxError ) ) );
					for ( const result of comparisons ) for ( const metric of Object.values( result ).filter( item => typeof item === 'object' ) ) {

						assert.equal( metric.maxError, 0, 'Classification and repeated controls preserve resolved/filtered RGBA exactly' );

					}

				}

			}

		}

	}
	const throughput = [];
	if ( process.env.CLASSIFY_SKIP_BENCHMARK !== '1' ) for ( const traversal of [ 'Linear', 'Conservative Hi-Z', 'Adaptive Hi-Z (experimental)' ] ) {

		for ( const classify of [ false, true, false ] ) {

			const result = await page.evaluate( async ( { traversal, classify } ) => {

				const s = window.__fallbackTest;
				await window.__drain();
				s.camera.position.copy( window.__cameraStart );
				s.controls.update();
				Object.assign( s.ssrParams, { traversal, classify, resolutionScale: 1, resolveOrder: 'Before filtering' } );
				s.updateIndirectLighting();
				s.updateSSRParameters();
				s.params.output = 'Beauty';
				s.updateOutput();
				window.__freezeLighting = false;
				const run = async duration => {

					let frames = 0;
					const before = s.ssrPass._noiseIndex.value;
					const start = performance.now();
					do {

						for ( let i = 0; i < 4; i ++ ) window.__step();
						await window.__drain();
						frames += 4;

					} while ( performance.now() - start < duration );
					const seconds = ( performance.now() - start ) / 1000;
					if ( s.ssrPass._noiseIndex.value - before !== frames ) throw new Error( 'SSR must execute once per completed frame' );
					return { frames, seconds, fps: frames / seconds };

				};
				await run( 1000 );
				await window.__drain();
				const result = await run( 5000 );
				await window.__drain();
				return result;

			}, { traversal, classify } );
			throughput.push( { traversal, classify, ...result } );
			console.log( 'Throughput', throughput.at( - 1 ) );
			await writeFile( `${ directory }/throughput.json`, JSON.stringify( { hardware, browser: await browser.version(), model, viewport: page.viewport(), throughput, errors }, null, 2 ) );

		}

	}
	assert.deepEqual( errors, [], 'No browser/GPU errors' );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

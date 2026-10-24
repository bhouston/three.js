/* global window, document, navigator, performance */
// Serial hardware comparison. Each manifest entry names an isolated worktree.
import puppeteer from 'puppeteer';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import console from 'node:console';
import { URL } from 'node:url';
import { createServer } from '../../utils/server.js';

const manifest = JSON.parse( await readFile( process.argv[ 2 ], 'utf8' ) );
const outputDirectory = process.env.SSGI_EXPERIMENT_OUTPUT || '/tmp/three-ssgi-experiments';
const referenceFrames = Number( process.env.SSGI_REFERENCE_FRAMES || 64 );
const measurementFrames = Number( process.env.SSGI_MEASUREMENT_FRAMES || 32 );
const geometryModule = await readFile( new URL( './ssgi-geometry-reference.js', import.meta.url ), 'utf8' );
let failedCases = 0;
await mkdir( outputDirectory, { recursive: true } );
const browser = await puppeteer.launch( {

	executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	headless: false,
	args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-background-timer-throttling', '--disable-renderer-backgrounding' ],
	defaultViewport: { width: 960, height: 720 }

} );
try {

	for ( const experiment of manifest ) {

		const root = path.resolve( experiment.root );
		const directory = path.join( outputDirectory, experiment.label );
		await mkdir( directory, { recursive: true } );
		const results = [];
		const server = createServer( { root } );
		await new Promise( resolve => server.listen( 0, resolve ) );
		try {

			for ( const model of experiment.models || [ 'Cornell Box', 'Dual Rooms' ] ) {

				const page = await browser.newPage();
				const errors = [];
				page.on( 'pageerror', error => { errors.push( error.message ); console.error( experiment.label, error.message ); } );
				page.on( 'console', message => { if ( message.type() === 'error' ) { errors.push( message.text() ); console.error( experiment.label, message.text().slice( 0, 1200 ) ); } } );
				try {

					let html = ( await readFile( path.join( root, 'examples/webgpu_interior.html' ), 'utf8' ) )
						.replace( "model: 'Dual Rooms'", `model: ${ JSON.stringify( model ) }` )
						.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
						.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
						.replace( 'renderer.setAnimationLoop( animate );', 'window.__experiment = { renderer, scene, model, camera, controls, params, ssgiParams, ssrParams, ssgiPass, ssgiDenoise, ssrPass, scenePass, prePass, lightingHistory, animate, updateOutput, updateSSGIParameters, updateSSRParameters, resetScreenSpaceLighting }; renderer.setAnimationLoop( animate );' );
					const animateStart = html.indexOf( 'function animate()' );
					const animateEnd = html.indexOf( 'function createHistorySource', animateStart );
					if ( animateStart < 0 || animateEnd < 0 ) throw new Error( 'Missing scene instrumentation anchors' );
					html = html.slice( 0, animateStart ) + html.slice( animateStart, animateEnd ).replaceAll( 'renderer.copyTextureToTexture(', 'if ( ! window.__freezeLighting ) renderer.copyTextureToTexture(' ) + html.slice( animateEnd );
					let shader = await readFile( path.join( root, 'examples/jsm/tsl/display/SSGINode.js' ), 'utf8' );
					shader = "import { hash, frameId } from 'three/tsl';\n" + shader
						.replace( 'this.useNoiseNode = uniform( true );', 'this.useNoiseNode = uniform( true ); this._experimentReference = uniform( false ); this._experimentGain = uniform( 1 );' )
						.replace( 'const sampleBeauty = ( uv ) => this.beautyNode.sample( uv );', 'const sampleBeauty = ( uv ) => { const value = this.beautyNode.sample( uv ); return vec4( value.rgb.mul( this._experimentGain ), value.a ); };' )
						.replace( 'const ao = float( 0 );', `If( this._experimentReference, () => {
							const seed = uint( screenCoordinate.x ).add( uint( screenCoordinate.y ).mul( 65537 ) ).add( frameId.mul( 1583458089 ) );
							noiseDirection.assign( hash( seed ) );
							initialRayStep.assign( hash( seed.add( 1013904223 ) ).add( hash( seed.add( 2027808446 ) ) ) );
						} );
						const ao = float( 0 );` );
					await page.setRequestInterception( true );
					page.on( 'request', request => {

						const pathname = new URL( request.url() ).pathname;
						if ( pathname.endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
						if ( pathname === '/examples/webgpu_interior.html' ) return request.respond( { status: 200, contentType: 'text/html', body: html } );
						if ( pathname.endsWith( '/ssgi-geometry-reference.js' ) ) return request.respond( { status: 200, contentType: 'text/javascript', body: geometryModule } );
						if ( pathname.endsWith( '/SSGINode.js' ) ) return request.respond( { status: 200, contentType: 'text/javascript', body: shader } );
						return request.continue();

					} );
					await page.goto( `http://localhost:${ server.address().port }/examples/webgpu_interior.html` );
					await page.waitForFunction( () => window.__experiment && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 180000 } );
					if ( errors.length ) throw new Error( 'Rendering errors while loading' );
					const hardware = await page.evaluate( async experiment => {

						const info = ( await navigator.gpu.requestAdapter() ).info;
						const hardware = { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
						if ( ! info.vendor || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( hardware ) ) ) throw new Error( 'Hardware GPU required' );
						const s = window.__experiment;
						s.renderer.setAnimationLoop( null ); s.renderer._animation.stop(); s.controls.enableDamping = false;
						Object.assign( s.ssgiParams, experiment.ssgi || {} );
						s.updateSSGIParameters();
						window.__step = () => {

							s.renderer.info.reset(); s.renderer._nodes.nodeFrame.update(); s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId; s.animate();

						};
						window.__drain = () => s.renderer.backend.device.queue.onSubmittedWorkDone();
						const { DataUtils } = await import( '/src/Three.WebGPU.js' );
						window.__read = async ( target, index = 0 ) => Float32Array.from( await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, index ), DataUtils.fromHalfFloat );
						for ( let i = 0; i < 64; i ++ ) window.__step();
						await window.__drain();
						return hardware;

					}, experiment );
					console.log( experiment.label, model, hardware );
					const settings = await page.evaluate( () => {

						const s = window.__experiment;
						return { ssgi: { ...s.ssgiParams }, worldRadius: s.ssgiPass.radius.value, worldThickness: s.ssgiPass.thickness.value };

					} );
					await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
					const slug = model.toLowerCase().replaceAll( ' ', '-' ).replaceAll( /[^a-z0-9-]/g, '' );
					for ( const output of experiment.captures === false ? [] : [ 'Beauty', 'SSGI Filtered', 'SSGI Filtered Coverage' ] ) {

						await page.evaluate( async output => {

							const s = window.__experiment; s.params.output = output; s.updateOutput();
							for ( let i = 0; i < 32; i ++ ) window.__step(); await window.__drain();

						}, output );
						await page.screenshot( { path: path.join( directory, `${ slug }-${ output.toLowerCase().replaceAll( ' ', '-' ) }.png` ) } );

					}
					const throughput = experiment.throughput === false ? null : await page.evaluate( async () => {

						const s = window.__experiment; s.params.output = 'Beauty'; s.updateOutput();
						for ( let i = 0; i < 64; i ++ ) window.__step(); await window.__drain();
						let completed = 0, draws = 0; const initial = s.ssgiDenoise._frames;
						const start = performance.now();
						while ( performance.now() - start < 5000 ) {

							for ( let i = 0; i < 4; i ++ ) { window.__step(); draws += s.renderer.info.render.drawCalls; }
							await window.__drain(); completed += 4;

						}
						const seconds = ( performance.now() - start ) / 1000;
						if ( s.ssgiDenoise._frames - initial !== completed ) throw new Error( 'Denoiser update count differs from completed frames' );
						return { completed, draws, seconds, fps: completed / seconds };

					} );
					const geometry = experiment.geometry === false ? null : await page.evaluate( async () => {

						const s = window.__experiment;
						await window.__drain();
						const { ssgiGeometryReference } = await import( '/test/benchmarks/ssgi-geometry-reference.js' );
						const target = s.ssgiPass.getRenderTarget();
						const raw = await window.__read( target, 1 );
						const filtered = await window.__read( s.ssgiDenoise.getRenderTarget() );
						const sourceTarget = s.ssgiPass.beautyNode.renderTarget;
						const radiance = sourceTarget ? await window.__read( sourceTarget ) : await window.__read( s.lightingHistory );
						return ssgiGeometryReference( { model: s.model, camera: s.camera, width: 960, height: 720, radius: s.ssgiPass.radius.value,
							radiance, raw: { data: raw, width: target.width, height: target.height }, filtered } );

					} );
					const parameterSweep = experiment.parameterSweep ? await page.evaluate( async () => {

						const s = window.__experiment;
						window.__freezeLighting = false;
						if ( s.ssgiPass._experimentReference ) s.ssgiPass._experimentReference.value = false;
						if ( s.ssgiPass._experimentGain ) s.ssgiPass._experimentGain.value = 1;

						const { runSSGIParameterSweep } = await import( '/test/benchmarks/ssgi-parameter-sweep.js' );
						return runSSGIParameterSweep( {
							state: s, step: window.__step, drain: window.__drain, read: window.__read,
							width: 960, height: 720,
							freezeLighting: value => { window.__freezeLighting = value; },
							getFreezeLighting: () => window.__freezeLighting
						} );

					} ) : null;
					let quality = null;
					if ( experiment.quality !== false ) quality = await page.evaluate( async ( { referenceFrames, measurementFrames, gainResponse } ) => {

						const s = window.__experiment;
						const saved = { ...s.ssgiParams };
						window.__freezeLighting = true;
						s.params.output = 'SSGI Filtered'; s.updateOutput();
						s.ssgiParams.resolutionScale = 1; s.ssgiParams.sliceCount = 8; s.ssgiParams.stepCount = 32; s.ssgiParams.maxLuminance = 1e6;
						s.updateSSGIParameters(); s.ssgiPass._experimentReference.value = true;
						for ( let i = 0; i < 16; i ++ ) window.__step(); await window.__drain();
						let reference;
						for ( let frame = 0; frame < referenceFrames; frame ++ ) {

							window.__step(); const value = await window.__read( s.ssgiPass.getRenderTarget(), 1 );
							reference ||= new Float64Array( value.length );
							for ( let i = 0; i < value.length; i += 4 ) {

								for ( let c = 0; c < 3; c ++ ) reference[ i + c ] += value[ i + c ] * value[ i + 3 ] / referenceFrames;
								reference[ i + 3 ] += value[ i + 3 ] / referenceFrames;

							}

						}
						Object.assign( s.ssgiParams, saved, { maxLuminance: 1e6 } ); s.updateSSGIParameters(); s.ssgiPass._experimentReference.value = false;
						s.renderer._nodes.nodeFrame.frameId = Math.ceil( s.renderer._nodes.nodeFrame.frameId / 192 ) * 192;
						for ( let i = 0; i < 64; i ++ ) window.__step(); await window.__drain();
						const sum = new Float64Array( reference.length ), squares = new Float64Array( reference.length );
						let mse = 0, coverageMse = 0;
						for ( let frame = 0; frame < measurementFrames; frame ++ ) {

							window.__step(); const value = await window.__read( s.ssgiDenoise.getRenderTarget() );
							if ( value.length !== reference.length ) throw new Error( 'Reference and reconstruction resolutions differ' );
							for ( let i = 0; i < value.length; i += 4 ) {

								for ( let c = 0; c < 4; c ++ ) {

									const v = c === 3 ? value[ i + c ] : value[ i + c ] * value[ i + 3 ];
									if ( ! Number.isFinite( v ) ) throw new Error( 'Nonfinite reconstruction' );
									sum[ i + c ] += v / measurementFrames; squares[ i + c ] += v * v / measurementFrames;
									if ( c === 3 ) coverageMse += ( v - reference[ i + c ] ) ** 2;
									else mse += ( v - reference[ i + c ] ) ** 2;

								}

							}

						}
						let variance = 0, meanMse = 0, coverageVariance = 0, meanCoverage = 0;
						for ( let i = 0; i < reference.length; i ++ ) {

							if ( i % 4 === 3 ) { coverageVariance += Math.max( 0, squares[ i ] - sum[ i ] ** 2 ); meanCoverage += sum[ i ]; }
							else { variance += Math.max( 0, squares[ i ] - sum[ i ] ** 2 ); meanMse += ( sum[ i ] - reference[ i ] ) ** 2; }

						}
						// Controlled source gain step, without rebaking or changing visibility.
						const response = [];
						if ( s.ssgiPass._experimentGain && gainResponse ) {

							s.ssgiPass._experimentGain.value = 2;
							for ( let frame = 1; frame <= 32; frame ++ ) {

								window.__step();
								if ( [ 1, 2, 4, 8, 16, 32 ].includes( frame ) ) {

									const value = await window.__read( s.ssgiDenoise.getRenderTarget() ); let error = 0;
									for ( let i = 0; i < value.length; i += 4 ) for ( let c = 0; c < 3; c ++ ) error += ( value[ i + c ] * value[ i + 3 ] - 2 * reference[ i + c ] ) ** 2;
									response.push( { frame, rmse: Math.sqrt( error / ( reference.length / 4 * 3 ) ) } );

								}

							}

						}
						const motionResponse = [];
						if ( s.ssgiPass._experimentGain ) {

							s.ssgiPass._experimentGain.value = 1; window.__freezeLighting = false;
							for ( let i = 0; i < 64; i ++ ) window.__step(); await window.__drain();
							const { Vector3 } = await import( '/src/Three.WebGPU.js' );
							s.camera.position.sub( s.controls.target ).applyAxisAngle( new Vector3( 0, 1, 0 ), 0.12 ).add( s.controls.target );
							s.controls.update(); window.__step(); await window.__drain(); window.__freezeLighting = true;
							const observations = [];
							for ( let frame = 1; frame <= 32; frame ++ ) {

								if ( frame > 1 ) window.__step();
								if ( [ 1, 2, 4, 8, 16, 32 ].includes( frame ) ) observations.push( { frame, value: await window.__read( s.ssgiDenoise.getRenderTarget() ) } );

							}
							s.ssgiParams.resolutionScale = 1; s.ssgiParams.sliceCount = 8; s.ssgiParams.stepCount = 32;
							s.updateSSGIParameters(); s.ssgiPass._experimentReference.value = true;
							for ( let i = 0; i < 16; i ++ ) window.__step(); await window.__drain();
							const motionReference = new Float64Array( reference.length );
							for ( let frame = 0; frame < referenceFrames; frame ++ ) {

								window.__step(); const value = await window.__read( s.ssgiPass.getRenderTarget(), 1 );
								for ( let i = 0; i < value.length; i += 4 ) for ( let c = 0; c < 3; c ++ ) motionReference[ i + c ] += value[ i + c ] * value[ i + 3 ] / referenceFrames;

							}
							for ( const { frame, value } of observations ) {

								let error = 0;
								for ( let i = 0; i < value.length; i += 4 ) for ( let c = 0; c < 3; c ++ ) error += ( value[ i + c ] * value[ i + 3 ] - motionReference[ i + c ] ) ** 2;
								motionResponse.push( { frame, rmse: Math.sqrt( error / ( reference.length / 4 * 3 ) ) } );

							}

						}
						await window.__drain();
						const pixels = reference.length / 4;
						return { referenceFrames, measurementFrames, reference: 'Independent hashed 8-slice 32-step full-resolution average; same estimator and frozen lighting, no luminance clamp', rmse: Math.sqrt( mse / ( pixels * 3 * measurementFrames ) ), meanRmse: Math.sqrt( meanMse / ( pixels * 3 ) ), temporalDeviation: Math.sqrt( variance / ( pixels * 3 ) ), coverageRmse: Math.sqrt( coverageMse / ( pixels * measurementFrames ) ), coverageDeviation: Math.sqrt( coverageVariance / pixels ), meanCoverage: meanCoverage / pixels, response, motionResponse };

					}, { referenceFrames, measurementFrames, gainResponse: experiment.gainResponse !== false } );
					if ( errors.length ) throw new Error( 'Rendering errors during measurements' );
					const commit = execFileSync( 'git-dedup', [ 'rev-parse', 'HEAD' ], { cwd: root, encoding: 'utf8' } ).trim();
					results.push( { model, commit, hardware, settings, throughput, geometry, parameterSweep, quality, errors } );
					console.log( experiment.label, model, JSON.stringify( { throughput, geometry: geometry?.comparisons, parameterSweep: parameterSweep?.length, quality } ) );

				} catch ( error ) {

					failedCases ++;
					results.push( { model, failed: error.message, errors } ); console.error( experiment.label, model, error.message );

				} finally {

					await page.close();
					await writeFile( path.join( directory, 'results.json' ), JSON.stringify( { experiment, viewport: [ 960, 720 ], lightMapResolution: 1024, results }, null, 2 ) );

				}

			}

		} finally {

			await new Promise( resolve => server.close( resolve ) );

		}

	}

} finally {

	await browser.close();

}

if ( failedCases ) throw new Error( `${ failedCases } experiment scenes failed; inspect the saved records` );

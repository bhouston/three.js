/* global window, document, navigator, performance */
// Run from the repository root. Requires hardware Chrome; never enables a software adapter.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';

const directory = process.env.SAMPLING_OUTPUT || '/tmp/three-interior-sampling';
const model = process.env.SAMPLING_MODEL || 'Primitives Floor (Color Rough)';
const frames = Number( process.env.SAMPLING_FRAMES || 96 );
const referenceFrames = Number( process.env.SAMPLING_REFERENCE_FRAMES || 256 );
const modes = ( process.env.SAMPLING_MODES || 'Original,FAST Product,FAST Separate' ).split( ',' );
const captureOnly = process.env.SAMPLING_CAPTURE_ONLY === '1';
if ( ! Number.isInteger( frames ) || frames < 2 || ! Number.isInteger( referenceFrames ) || referenceFrames < 2 ) throw new Error( 'At least two measurement and reference frames are required' );
if ( modes.some( mode => ! [ 'Original', 'FAST Product', 'FAST Separate', 'Selected' ].includes( mode ) ) ) throw new Error( 'Unsupported sampling mode' );
const swizzle = process.env.SAMPLING_SWIZZLE || 'wxy';
if ( ! [ 'xyz', 'wyz', 'wxy' ].includes( swizzle ) ) throw new Error( 'Unsupported noise swizzle' );
const atlasDirectory = process.env.SAMPLING_ATLAS_DIRECTORY;
await mkdir( directory, { recursive: true } );
const html = ( await readFile( 'examples/webgpu_interior.html', 'utf8' ) )
	.replace( 'new THREE.Vector2( 73, 29 ) ).wxy', `new THREE.Vector2( 73, 29 ) ).${ swizzle }` )
	.replace( "sampling: 'FAST Separate'", "sampling: 'Original'" )
	.replace( "model: 'Dual Rooms'", `model: ${ JSON.stringify( model ) }` )
	.replace( "output: 'Beauty'", "output: 'SSR Filtered'" )
	.replace( "const samplingModes = [ 'Original', 'FAST Product', 'FAST Separate' ];", "const samplingModes = [ 'Original', 'FAST Product', 'FAST Separate', 'Reference' ];" )
	.replace( 'import { Fn, If,', 'import { hash, uint, frameId, screenCoordinate, Fn, If,' )
	.replace( 'If( mode.equal( 2 ),', `const seed = uint( screenCoordinate.x ).add( uint( screenCoordinate.y ).mul( 65537 ) ).add( frameId.mul( 1583458089 ) ).add( offset.x * 9277 + offset.y * 26699 );
		If( mode.equal( 3 ), () => value.assign( vec4( hash( seed ), hash( seed.add( 1013904223 ) ), hash( seed.add( 2027808446 ) ), hash( seed.add( 3041712669 ) ) ) ) ).ElseIf( mode.equal( 2 ),` )
	.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
	.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
	.replace( "renderer.copyTextureToTexture( scenePass.getTexture( 'output' ), lightingHistory.texture );", "if ( ! window.__freezeLighting ) renderer.copyTextureToTexture( scenePass.getTexture( 'output' ), lightingHistory.texture );" )
	.replace( 'renderer.setAnimationLoop( animate );', `window.__sampling = { renderer, camera, controls, params, ssrParams, ssgiParams, ssrPass, ssgiPass, ssrCoverageDenoise, ssgiDenoise, ssrDenoise, scenePass, prePass, lightingHistory, animate, updateOutput, updateSSRParameters, updateSSGIParameters, resetScreenSpaceLighting }; renderer.setAnimationLoop( animate );` );
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-background-timer-throttling', '--disable-renderer-backgrounding' ],
		defaultViewport: { width: 960, height: 720 }
	} );
	if ( process.env.SAMPLING_UNIT_ONLY === '1' ) {

		const page = await browser.newPage();
		await page.goto( `http://localhost:${ server.address().port }/` );
		const adapter = await page.evaluate( async () => {

			const info = ( await navigator.gpu.requestAdapter() ).info;
			return { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter };

		} );
		if ( ! adapter.vendor || adapter.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( adapter ) ) ) throw new Error( 'Hardware GPU required' );
		console.log( adapter );
		const results = [];
		for ( const filter of [ 'FASTNoise', 'SSGINode', 'CoverageDenoiseNode', 'TemporalPipeline' ] ) {

			await page.goto( `http://localhost:${ server.address().port }/test/unit/UnitTestsAddons.html?filter=${ filter }` );
			await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 120000 } );
			const result = await page.evaluate( () => ( { stats: window._QUnitStats, failures: Array.from( document.querySelectorAll( '#qunit-tests > .fail' ), element => element.textContent ), skipped: document.getElementById( 'qunit-tests' ).textContent.includes( 'SKIPPED' ) } ) );
			results.push( { filter, ...result } );
			console.log( filter, result );
			if ( result.stats.failed || ! result.stats.passed || result.skipped ) throw new Error( `${ filter }: tests failed or GPU tests skipped` );

		}
		await writeFile( `${ directory }/unit-tests.json`, JSON.stringify( { adapter, results }, null, 2 ) );

	} else {

		const page = await browser.newPage();
		const errors = [];
		page.on( 'pageerror', error => { errors.push( error.message ); console.error( error.message ); } );
		page.on( 'console', message => { if ( message.type() === 'error' ) { errors.push( message.text() ); console.error( message.text().slice( 0, 1500 ) ); } } );
		await page.setRequestInterception( true );
		page.on( 'request', async request => {

			const atlasName = request.url().match( /gauss1-ema01-(product|separate05)\.png$/ );
			if ( atlasDirectory && atlasName ) return request.respond( { status: 200, contentType: 'image/png', body: await readFile( `${ atlasDirectory }/gauss1-ema01-${ atlasName[ 1 ] }.png` ) } );
			return request.url().endsWith( '/favicon.ico' ) ? request.respond( { status: 204 } ) : request.url().split( '?' )[ 0 ].endsWith( '/examples/webgpu_interior.html' ) ? request.respond( { status: 200, contentType: 'text/html', body: html } ) : request.continue();

		} );
		await page.goto( `http://localhost:${ server.address().port }/examples/webgpu_interior.html` );
		await page.waitForFunction( () => window.__sampling && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 240000 } );
		console.log( 'Bake complete', model );
		await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
		const hardware = await page.evaluate( async () => {

			const s = window.__sampling;
			const adapter = await navigator.gpu.requestAdapter();
			const info = adapter.info;
			const result = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
			if ( info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( result ) ) || ! info.vendor ) throw new Error( 'A verified hardware GPU is required' );
			s.renderer.setAnimationLoop( null );
			s.renderer._animation.stop();
			s.controls.enableDamping = false;
			const { DataUtils } = await import( '/src/Three.WebGPU.js' );
			const queue = s.renderer.backend.device.queue;
			window.__step = () => { s.renderer.info.reset(); s.renderer._nodes.nodeFrame.update(); s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId; s.animate(); };
			window.__drain = () => queue.onSubmittedWorkDone();
			for ( let i = 0; i < 64; i ++ ) window.__step();
			await window.__drain();
			window.__freezeLighting = true;
			window.__frozenLighting = s.lightingHistory.clone();
			s.renderer.initRenderTarget( window.__frozenLighting );
			s.renderer.copyTextureToTexture( s.lightingHistory.texture, window.__frozenLighting.texture );
			window.__read = async ( target, index = 0 ) => {

				const data = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, index );
				return Float32Array.from( data, DataUtils.fromHalfFloat );

			};
			window.__runs = [];
			return result;

		} );
		console.log( hardware );
		for ( const mode of [ ...modes, ... ( captureOnly ? [] : [ 'Reference' ] ) ] ) {

			await page.evaluate( async mode => {

				const s = window.__sampling;
				await window.__drain();
				s.ssrParams.sampling = mode === 'Selected' ? 'FAST Separate' : mode;
				s.ssgiParams.sampling = mode === 'Selected' ? 'Original' : mode;
				s.updateSSRParameters(); s.updateSSGIParameters();
				// Keep the original SSR index and SSGI temporal cycle aligned between methods.
				s.ssrPass._noiseIndex.value = 0;
				s.renderer._nodes.nodeFrame.frameId = Math.ceil( s.renderer._nodes.nodeFrame.frameId / 96 ) * 96;
				// Restore source validity after resetting denoiser history without changing frozen RGB.
				window.__step();
				for ( let i = 0; i < 64; i ++ ) window.__step();
				await window.__drain();

			}, mode );
			const slug = mode.toLowerCase().replaceAll( ' ', '-' );
			if ( mode !== 'Reference' && process.env.SAMPLING_SKIP_CAPTURES !== '1' ) {

				for ( const view of [ 'SSR Raw', 'SSR Filtered', 'SSR Coverage', 'SSR Filtered Coverage', 'SSGI Raw', 'SSGI Filtered', 'SSGI Coverage', 'SSGI Filtered Coverage', 'Beauty' ] ) {

					await page.evaluate( async view => {

						const s = window.__sampling;
						s.params.output = view; s.updateOutput();
						for ( let i = 0; i < 24; i ++ ) window.__step();
						await window.__drain();

					}, view );
					await page.screenshot( { path: `${ directory }/${ slug }-${ view.toLowerCase().replaceAll( ' ', '-' ) }.png` } );

				}

			}
			if ( captureOnly ) continue;

			const measurement = await page.evaluate( async ( { mode, frames } ) => {

				const s = window.__sampling;
				s.params.output = 'SSR Filtered'; s.updateOutput();
				s.updateSSRParameters(); s.updateSSGIParameters();
				s.ssrPass._noiseIndex.value = 0;
				s.renderer._nodes.nodeFrame.frameId = Math.ceil( s.renderer._nodes.nodeFrame.frameId / 96 ) * 96;
				for ( let i = 0; i < 64; i ++ ) window.__step();
				await window.__drain();
				const targets = {
					ssrRaw: [ s.ssrCoverageDenoise.inputNode.renderTarget, 0, true ],
					ssrFiltered: [ s.ssrCoverageDenoise.getRenderTarget(), 0, true ],
					ssgiRaw: [ s.ssgiPass.getRenderTarget(), 1, true ],
					ssgiFiltered: [ s.ssgiDenoise.getRenderTarget(), 0, true ]
				};
				if ( mode === 'Reference' ) { delete targets.ssrFiltered; delete targets.ssgiFiltered; }
				const stats = {};
				for ( let frame = 0; frame < frames; frame ++ ) {

					window.__step();
					for ( const [ name, [ target, index, premultiply ] ] of Object.entries( targets ) ) {

						const data = await window.__read( target, index );
						if ( premultiply ) for ( let i = 0; i < data.length; i += 4 ) for ( let c = 0; c < 3; c ++ ) data[ i + c ] *= data[ i + 3 ];
						if ( ! stats[ name ] ) stats[ name ] = { width: target.width, height: target.height, sum: new Float64Array( data.length ), square: new Float64Array( data.length ), delta: new Float64Array( 4 ), previous: data };
						const t = stats[ name ];
						for ( let i = 0; i < data.length; i ++ ) {

							if ( ! Number.isFinite( data[ i ] ) ) throw new Error( `${ name }: non-finite output` );
							t.sum[ i ] += data[ i ]; t.square[ i ] += data[ i ] ** 2;
							if ( frame ) t.delta[ i % 4 ] += ( data[ i ] - t.previous[ i ] ) ** 2;

						}
						t.previous = data;

					}

				}
				window.__runs.push( { mode, frames, stats } );
				return Object.fromEntries( Object.entries( stats ).map( ( [ name, t ] ) => [ name, { width: t.width, height: t.height } ] ) );

			}, { mode, frames: mode === 'Reference' ? referenceFrames : frames } );
			console.log( 'Quality captured', mode, measurement );
			if ( mode === 'Reference' || process.env.SAMPLING_SKIP_BENCHMARK === '1' ) continue;
			// Benchmark full beauty with live history copies; no readbacks or screenshots in the timed loop.
			const throughput = await page.evaluate( async () => {

				const s = window.__sampling;
				s.params.output = 'Beauty'; s.updateOutput();
				window.__freezeLighting = false;
				for ( let i = 0; i < 64; i ++ ) window.__step();
				await window.__drain();
				const samples = [];
				for ( let repeat = 0; repeat < 3; repeat ++ ) {

					let completed = 0, drawCalls = 0, effectFrames = 0;
					const initialEffectFrame = s.ssgiDenoise._frames;
					const initialSSRFrame = s.ssrPass._noiseIndex.value;
					const start = performance.now();
					while ( performance.now() - start < 5000 ) {

						// Drain a batch, then count only GPU-completed full rendering frames.
						for ( let batch = 0; batch < 4; batch ++ ) { window.__step(); drawCalls += s.renderer.info.render.drawCalls; }
						await window.__drain(); completed += 4;

					}
					const seconds = ( performance.now() - start ) / 1000;
					effectFrames = s.ssgiDenoise._frames - initialEffectFrame;
					const ssrFrames = s.ssrPass._noiseIndex.value - initialSSRFrame;
					if ( effectFrames !== completed || ssrFrames !== completed ) throw new Error( 'Not all effect frames completed' );
					samples.push( { completed, effectFrames, ssrFrames, drawCalls, seconds, fps: completed / seconds } );

				}
				window.__freezeLighting = true;
				s.renderer.copyTextureToTexture( window.__frozenLighting.texture, s.lightingHistory.texture );
				await window.__drain();
				return samples;

			} );
			await writeFile( `${ directory }/${ slug }-throughput.json`, JSON.stringify( throughput, null, 2 ) );
			console.log( 'Throughput', mode, throughput );

		}
		if ( ! captureOnly ) await writeFile( `${ directory }/quality.json`, JSON.stringify( await page.evaluate( () => {

			const runs = window.__runs;
			// Independent PCG reference uses the same integrators and frozen source lighting.
			const reference = {};
			for ( const name of [ 'ssrRaw', 'ssgiRaw' ] ) {

				const first = runs[ 0 ].stats[ name ];
				const data = new Float64Array( first.sum.length );
				let count = 0;
				for ( const run of runs.filter( r => r.mode === 'Reference' ) ) { count += run.frames; for ( let i = 0; i < data.length; i ++ ) data[ i ] += run.stats[ name ].sum[ i ]; }
				for ( let i = 0; i < data.length; i ++ ) data[ i ] /= count;
				reference[ name ] = { data, width: first.width, height: first.height };

			}
			return runs.map( run => ( { mode: run.mode, frames: run.frames, signals: Object.fromEntries( Object.entries( run.stats ).map( ( [ name, t ] ) => {

				const ref = reference[ name.startsWith( 'ssgi' ) ? 'ssgiRaw' : 'ssrRaw' ];
				let error = 0, variance = 0, mean = 0, coverageMean = 0, coverageError = 0, coverageVariance = 0;
				for ( let y = 0; y < t.height; y ++ ) for ( let x = 0; x < t.width; x ++ ) {

					const i = 4 * ( y * t.width + x );
					const r = 4 * ( Math.floor( ( y + 0.5 ) * ref.height / t.height ) * ref.width + Math.floor( ( x + 0.5 ) * ref.width / t.width ) );
					for ( let c = 0; c < 4; c ++ ) {

						const m = t.sum[ i + c ] / run.frames;
						const v = Math.max( 0, t.square[ i + c ] / run.frames - m * m );
						const e = v + ( m - ref.data[ r + c ] ) ** 2;
						if ( c < 3 ) { error += e; variance += v; mean += m; }
						else { coverageMean += m; coverageError += e; coverageVariance += v; }

					}

				}
				const pixels = t.width * t.height;
				return [ name, { meanRGB: mean / ( pixels * 3 ), rmseRGB: Math.sqrt( error / ( pixels * 3 ) ), temporalStdRGB: Math.sqrt( variance / ( pixels * 3 ) ), frameDeltaRGB: Math.sqrt( ( t.delta[ 0 ] + t.delta[ 1 ] + t.delta[ 2 ] ) / ( pixels * 3 * ( run.frames - 1 ) ) ), meanAlpha: coverageMean / pixels, rmseAlpha: Math.sqrt( coverageError / pixels ), temporalStdAlpha: Math.sqrt( coverageVariance / pixels ) } ];

			} ) ) } ) );

		} ), null, 2 ) );
		await writeFile( `${ directory }/metadata.json`, JSON.stringify( { model, frames, referenceFrames, swizzle, atlasDirectory, captureOnly, hardware, errors, browserVersion: await browser.version(), viewport: { width: 960, height: 720 }, lightMapResolution: 1024 }, null, 2 ) );
		if ( errors.length ) throw new Error( `${ errors.length } browser errors; see metadata.json` );

	}

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
// Run from repository root: node test/webgpu/benchmarkSSRHiZ.js <output.json> [baseline HiZNode.js]
const [ output, baseline ] = process.argv.slice( 2 );
if ( ! output ) throw new Error( 'An output JSON path is required.' );
const server = createServer();
await new Promise( r=>server.listen( 8081, r ) );
const browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || ( fs.existsSync( 'C:/Program Files/Google/Chrome/Application/chrome.exe' ) ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : undefined ), headless: false, args: [ '--disable-frame-rate-limit', '--disable-gpu-vsync', '--no-sandbox' ] } );
const results = [];
const width = Number( process.env.BENCH_WIDTH || 801 ), height = Number( process.env.BENCH_HEIGHT || 603 );
const modes = process.env.BENCH_MODES ? process.env.BENCH_MODES.split( ',' ) : [ 'linear', 'build-only', 'hi-z', 'frozen-hi-z' ];
const qualities = process.env.BENCH_QUALITY ? [ Number( process.env.BENCH_QUALITY ) ] : [ 0.5, 1 ];
try {

	for ( const scene of [ 'webgpu_postprocessing_ssr', 'webgpu_postprocessing_ssr_denoise' ] ) {

		const page = await browser.newPage();
		await page.setViewport( { width, height, deviceScaleFactor: 1 } );
		page.on( 'pageerror', e=>console.error( e.message ) );
		await page.setRequestInterception( true );
		page.on( 'request', async req=>{

			if ( req.url().endsWith( '/' + scene + '.html' ) ) {

				let html = fs.readFileSync( 'examples/' + scene + '.html', 'utf8' );
				const denoise = scene.endsWith( 'denoise' );
				html = html.replace( 'updateOutputNode();', `updateOutputNode(); window.__bench = { renderer, pipeline: ${denoise ? 'postProcessing' : 'renderPipeline'}, reflection: ${denoise ? 'ssrNode' : 'ssrPass'}, pyramid: depthPyramid, params, updateOutputNode, get model() { return model; } };` );
				await req.respond( { status: 200, contentType: 'text/html', body: html } );

			} else if ( baseline && req.url().endsWith( '/HiZNode.js' ) ) await req.respond( { status: 200, contentType: 'application/javascript', body: fs.readFileSync( baseline, 'utf8' ) } );
			else if ( baseline && req.url().endsWith( '/SSRNode.js' ) ) await req.respond( { status: 200, contentType: 'application/javascript', body: fs.readFileSync( path.join( path.dirname( baseline ), 'SSRNode-baseline.js' ), 'utf8' ) } );
			else await req.continue();

		} );
		await page.goto( 'http://localhost:8081/examples/' + scene + '.html' );
		await page.waitForFunction( ()=>window.__bench?.model, { timeout: 180000 } );
		const adapter = await page.evaluate( async()=>{

			const b = window.__bench;
			b.renderer.setAnimationLoop( null );
			b.errors = [];
			b.renderer.backend.device.addEventListener( 'uncapturederror', event => b.errors.push( event.error.message ) );
			await b.renderer.backend.device.queue.onSubmittedWorkDone();
			const a = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
			const info = b.renderer.backend.device.adapterInfo || a.info;
			const description = [ info.vendor, info.architecture, info.device, info.description ].join( ' ' );
			if ( ! description.trim() || a.isFallbackAdapter || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software|microsoft basic render/i.test( description ) ) throw new Error( 'Physical GPU required: ' + description );
			return description;

		} );
		for ( const quality of qualities ) for ( const mode of modes ) {

			const result = await page.evaluate( async( { quality, mode } )=>{

				const b = window.__bench;
				const device = b.renderer.backend.device;
				b.reflection.quality.value = quality;
				b.params.hiZ = mode !== 'linear';
				b.params.hiZTracing = [ 'hi-z', 'frozen-hi-z', 'adaptive', 'adaptive-coarse' ].includes( mode );
				b.params.hiZTraversal = mode.startsWith( 'adaptive' ) ? 'adaptive' : 'conservative';
				b.params.hiZMinMip = mode === 'adaptive-coarse' ? 1 : 0;
				b.updateOutputNode();
				// Freeze only after rendering once with a fresh pyramid.
				const draw = ()=>{

					b.renderer._nodes.nodeFrame.update();
					b.pipeline.render();

				};

				draw();
				await device.queue.onSubmittedWorkDone();
				if ( mode === 'frozen-hi-z' ) b.pyramid.enabled = false;
				const run = async duration=>{

					let frames = 0;
					const start = performance.now();
					do {

						for ( let i = 0; i < 8; i ++ ) {

							draw();
							frames ++;

						}

						await device.queue.onSubmittedWorkDone();

					} while ( performance.now() - start < duration );

					return { frames, ms: performance.now() - start };

				};

				await run( 1000 );
				await device.queue.onSubmittedWorkDone();
				const before = b.reflection._noiseIndex.value;
				const computeBefore = b.renderer.info.compute.calls;
				const r = await run( 5000 );
				await device.queue.onSubmittedWorkDone();
				if ( b.errors.length ) throw new Error( b.errors.join( '\n' ) );
				if ( b.reflection._noiseIndex.value - before !== r.frames ) throw new Error( 'SSR did not execute once per frame.' );
				const computeCalls = b.renderer.info.compute.calls - computeBefore;
				if ( ( mode === 'linear' || mode === 'frozen-hi-z' ) ? computeCalls !== 0 : computeCalls < r.frames ) throw new Error( 'Unexpected Hi-Z dispatch count.' );
				return { ...r, frameMs: r.ms / r.frames, fps: r.frames * 1000 / r.ms, noiseBefore: before, noiseAfter: b.reflection._noiseIndex.value, computeCalls };

			}, { quality, mode } );
			const row = { scene, adapter, quality, mode, ...result };
			results.push( row );
			console.log( JSON.stringify( row ) );
			fs.writeFileSync( output, JSON.stringify( { viewport: [ width, height ], baseline: baseline || null, results }, null, 2 ) );

		}

		// Controls are changed programmatically; hide their stale DOM values in view captures.
		await page.evaluate( () => {

			for ( const element of document.body.children ) if ( element.tagName !== 'CANVAS' ) element.style.visibility = 'hidden';

		} );
		await page.screenshot( { path: path.join( path.dirname( output ), scene + '-' + ( baseline ? 'baseline' : 'optimized' ) + '.png' ) } );
		await page.close();

	}

} finally {

	await browser.close();
	await new Promise( r=>server.close( r ) );

}

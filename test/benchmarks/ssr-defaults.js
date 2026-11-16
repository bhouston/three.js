// Full-pipeline sustained throughput comparison. No generated build files needed.
// node test/benchmarks/ssr-defaults.js <baseline-root> <output-json>
import puppeteer from 'puppeteer';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { createServer } from '../../utils/server.js';

const currentRoot = fileURLToPath( new URL( '../../', import.meta.url ) );
const [ baselineInput, output ] = process.argv.slice( 2 );
const baselineRoot = baselineInput && resolve( baselineInput );
if ( ! baselineRoot || ! output ) throw new Error( 'Supply baseline root and output JSON path.' );
const results = [];
let browser;
const servers = [];
try {

	for ( const root of [ baselineRoot, currentRoot ] ) {

		const server = createServer( { root } );
		await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
		servers.push( server );

	}

	browser = await puppeteer.launch( { executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--disable-frame-rate-limit', '--disable-gpu-vsync', '--disable-software-rasterizer' ] } );
	for ( const example of [ 'mirror', 'stochastic', 'animation' ] ) {

		for ( const mode of [ 'original', 'adaptive', 'linear', 'quality' ] ) {

			const original = mode === 'original';
			const root = original ? baselineRoot : currentRoot;
			const port = servers[ original ? 0 : 1 ].address().port;
			const page = await browser.newPage();
			const errors = [];
			page.on( 'pageerror', error => errors.push( error.message ) );
			page.on( 'requestfailed', request => errors.push( `Request failed: ${ request.url() }` ) );
			page.on( 'response', response => {

				if ( response.status() >= 400 && ! response.url().endsWith( '/favicon.ico' ) ) errors.push( `HTTP ${ response.status() }: ${ response.url() }` );

			} );
			page.on( 'console', message => {

				if ( /GPUValidationError|THREE\.TSL:.*error/i.test( message.text() ) ) errors.push( message.text() );

			} );
			await page.setViewport( { width: 1920, height: 1080, deviceScaleFactor: 1 } );
			// Repeatable random inputs for temporal sampling and scene construction.
			await page.evaluateOnNewDocument( () => {

				let seed = 193; Math.random = () => {

					seed = ( Math.imul( seed, 1664525 ) + 1013904223 ) >>> 0; return seed / 4294967296;

				};

			} );
			const filename = `webgpu_postprocessing_ssr_${ example }.html`;
			let html = await readFile( `${ root }/examples/${ filename }`, 'utf8' );
			if ( ! html.includes( 'renderer.setAnimationLoop( animate );' ) ) throw new Error( 'Missing animation-loop marker.' );
			html = html.replaceAll( '../build/three.webgpu.js', '../src/Three.WebGPU.js' ).replaceAll( '../build/three.tsl.js', '../src/Three.TSL.js' );
			html = html.replaceAll( 'renderer.setAnimationLoop( animate );', '' );
			// Drive animation at a repeatable simulated 60 Hz, independent of speed.
			html = html.replace( 'const delta = Math.min( timer.getDelta(), 0.05 );', 'const delta = 1 / 60;' );
			// Every batch renders the same eight poses over six simulated seconds.
			// Keep default animation, controls, temporal passes, and all draw work.
			html = html.replace( 'if ( params.animate ) time += delta;', 'if ( params.animate ) time = ( window.__benchmarkFrame ++ % 8 ) * 0.75;' );
			const params = example === 'stochastic' ? 'params.ssr' : 'params';
			const apply = example === 'stochastic' ? 'applyParams' : 'updateParameters';
			const marker = example === 'stochastic' ? 'updateOutputNode();' : 'updateOutput();';
			if ( ! html.includes( marker ) ) throw new Error( 'Missing initialization marker.' );
			const ready = example === 'animation' ? 'true' : 'model !== undefined';
			html = html.replace( marker, `${ marker }\nwindow.__benchmarkFrame = 0; window.__bench = { renderer, params: ${ params }, apply: ${ apply }, animate, ready: () => ${ ready } };` );
			await page.setRequestInterception( true );
			page.on( 'request', request => {

				if ( new URL( request.url() ).pathname === `/examples/${ filename }` ) request.respond( { contentType: 'text/html', body: html } );
				else request.continue();

			} );
			await page.goto( `http://127.0.0.1:${ port }/examples/${ filename }` );
			await page.waitForFunction( () => window.__bench?.ready(), { timeout: 120000 } );
			await page.waitForNetworkIdle( { idleTime: 1000, timeout: 120000 } );
			if ( errors.length ) throw new Error( errors.join( '\n' ) );
			const result = await page.evaluate( async mode => {

				const { renderer, params, apply, animate } = window.__bench;
				if ( ! renderer.backend.isWebGPUBackend ) throw new Error( 'Hardware WebGPU required.' );
				const device = renderer.backend.device;
				const info = device.adapterInfo;
				const adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
				if ( adapter.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( adapter ) ) || ! /nvidia|amd|intel|apple|qualcomm/i.test( JSON.stringify( adapter ) ) ) throw new Error( `Physical GPU required: ${ JSON.stringify( adapter ) }` );
				if ( mode !== 'original' ) params.adaptive = mode === 'adaptive';
				if ( mode === 'quality' ) Object.assign( params, { quality: 1, maxDistance: 50, binaryRefine: true } );
				apply();
				device.pushErrorScope( 'validation' );
				const batch = async () => {

					for ( let i = 0; i < 8; i ++ ) {

						// Advance FRAME nodes even outside the renderer's animation loop.
						renderer._nodes.nodeFrame.update();
						animate();

					}
					await device.queue.onSubmittedWorkDone();

				};

				const warmStart = performance.now();
				do {

					await batch();

				} while ( performance.now() - warmStart < 2000 );

				const trials = [];
				for ( let trial = 0; trial < 3; trial ++ ) {

					await device.queue.onSubmittedWorkDone();
					let completed = 0;
					const start = performance.now();
					do {

						await batch(); completed += 8;

					} while ( performance.now() - start < 5000 );

					trials.push( { completed, elapsedMs: performance.now() - start } );

				}

				const validation = await device.popErrorScope();
				if ( validation ) throw new Error( validation.message );
				return { adapter, settings: JSON.parse( JSON.stringify( params ) ), trials, averageMs: trials.reduce( ( sum, trial ) => sum + trial.elapsedMs, 0 ) / trials.reduce( ( sum, trial ) => sum + trial.completed, 0 ) };

			}, mode );
			if ( errors.length ) throw new Error( errors.join( '\n' ) );
			results.push( { example, mode, ...result } );
			await writeFile( output, JSON.stringify( { chrome: await browser.version(), width: 1920, height: 1080, results }, null, 2 ) );
			console.log( `${ example } ${ mode }: ${ result.averageMs.toFixed( 3 ) } ms` );
			await page.screenshot( { path: output.replace( /\.json$/, `-${ example }-${ mode }.png` ) } );
			await page.close();

		}

	}

} finally {

	await browser?.close();
	for ( const server of servers ) await new Promise( resolve => server.close( resolve ) );

}

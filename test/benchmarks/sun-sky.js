// Run from the repository root. Optional arguments: example HTML, output JSON.
// The HTML override allows comparison with an unmodified revision without a rebuild.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import fs from 'node:fs/promises';
const server = createServer();
await new Promise( r => server.listen( 8080, r ) );
const browser = await puppeteer.launch( { headless: false, args: [ '--enable-unsafe-webgpu', '--disable-frame-rate-limit', '--disable-gpu-vsync', '--disable-software-rasterizer' ], defaultViewport: { width: 1000, height: 700 } } );
try {

	const page = await browser.newPage();
	page.on( 'pageerror', e=>console.log( 'ERROR', e.message ) );
	await page.setRequestInterception( true );
	page.on( 'request', async r=>{

		if ( r.url().endsWith( '/examples/webgpu_sky.html' ) ) {

			let html = await fs.readFile( process.argv[ 2 ] || 'examples/webgpu_sky.html', 'utf8' );
			html = html.replace( 'autoRotate: true', 'autoRotate: false' );
			html = html.replace( 'const start = performance.now();', 'const start = performance.now(); window.firstBakeStart ??= start;' );
			html = html.replace( 'if ( ! baking ) renderer.render( scene, camera );', `if ( ! baking ) {
				renderer.render( scene, camera );
				window.firstComplete ??= renderer.backend.device.queue.onSubmittedWorkDone().then( () => performance.now() - window.firstBakeStart );
			}` );
			html = html.replace( 'renderer.setAnimationLoop( animate );', 'window.bench = { renderer, scene, camera, params, bakeSky, changed, animate, renderPass, skyMaterial, skyTarget, meterMaterial, meterTarget, pmrem, get dirty() { return dirty; }, get environment() { return environment; }, stop() { renderer.setAnimationLoop(null); }, get baking() { return baking; } }; renderer.setAnimationLoop( animate );' );
			await r.respond( { status: 200, contentType: 'text/html', body: html } );

		} else r.continue();

	} );
	await page.goto( 'http://localhost:8080/examples/webgpu_sky.html', { timeout: 120000 } );
	await page.waitForFunction( 'window.firstComplete && window.bench && !bench.baking && bench.environment', { timeout: 120000 } );
	const result = await page.evaluate( async()=>{

		const b = window.bench;
		const coldCompleteMs = await window.firstComplete;
		b.stop();
		const info = b.renderer.backend.device.adapterInfo;
		if ( info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( [ info.vendor, info.architecture, info.description ].join( ' ' ) ) || ! /nvidia|amd|intel|apple|qualcomm/i.test( info.vendor ) ) throw Error( 'Hardware GPU verification blocked' );
		const drain = ()=>b.renderer.backend.device.queue.onSubmittedWorkDone();
		await drain();
		const run = async( kind )=>{

			const step = async()=>{

				if ( kind === 'sky' ) b.renderPass( b.skyMaterial, b.skyTarget );
				else if ( kind === 'pmrem' ) b.pmrem.fromEquirectangular( b.skyTarget.texture, b.environment );
				else {

					b.params.altitude = 20 + Math.sin( n ) * 10;
					if ( kind === 'interactive' ) {

						b.changed();
						do {

							b.animate(); await new Promise( r=>setTimeout( r, 0 ) );

						} while ( b.dirty || b.baking );

					} else await b.bakeSky();
					b.renderer.render( b.scene, b.camera );

				}

				await drain(); n ++;

			};

			let n = 0; for ( let i = 0; i < 3; i ++ ) await step(); await drain(); n = 0;
			const start = performance.now(); do {

				await step();

			} while ( performance.now() - start < 5000 );

			const ms = performance.now() - start; return { frames: n, ms, msPerUpdate: ms / n, updatesPerSecond: n * 1000 / ms };

		};

		const timings = { adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description }, coldCompleteMs, sky: await run( 'sky' ), pmrem: await run( 'pmrem' ), complete: await run( 'complete' ), interactive: await run( 'interactive' ) };
		// Fixed final state for pixel comparisons, including all six PMREM faces.
		b.params.altitude = 32; await b.bakeSky(); b.renderer.render( b.scene, b.camera ); await drain();
		window.referencePixels = { sky: Array.from( await b.renderer.readRenderTargetPixelsAsync( b.skyTarget, 0, 0, b.skyTarget.width, b.skyTarget.height ) ), environment: [] };
		for ( let face = 0; face < 6; face ++ ) window.referencePixels.environment.push( Array.from( await b.renderer.readRenderTargetPixelsAsync( b.environment, 0, 0, b.environment.width, b.environment.height, 0, face ) ) );
		return timings;

	} );
	console.log( JSON.stringify( result, null, 2 ) );
	const output = process.argv[ 3 ] || 'sun-sky-benchmark.json';
	await fs.writeFile( output, JSON.stringify( result, null, 2 ) );
	await fs.writeFile( output.replace( '.json', '-pixels.json' ), JSON.stringify( await page.evaluate( ()=>window.referencePixels ) ) );
	await page.screenshot( { path: output.replace( '.json', '.png' ) } );

} finally {

	await browser.close(); await new Promise( r=>server.close( r ) );

}


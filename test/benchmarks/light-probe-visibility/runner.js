import { patch, patchBake } from './variants.js';
import puppeteer from 'puppeteer';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createServer } from '../../../utils/server.js';

const output = path.resolve( process.env.PROBE_RESULTS || path.join( os.homedir(), 'Coding/.benchmarks/three-light-probe-visibility' ) );
await mkdir( output, { recursive: true } );
const root = process.cwd();
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const url = `http://127.0.0.1:${server.address().port}`;
const variants = process.argv.slice( 2 ).length ? process.argv.slice( 2 ) : [
	'baseline', 'visibility-off', 'slack-0', 'slack-0.005', 'slack-0.01',
	'crush', 'distance-32', 'normal-0', 'normal-0.02', 'normal-0.16',
	'view-0', 'view-0.08', 'view-0.16', 'exact', 'region-mask', 'direct-convolution'
];

const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit' ], defaultViewport: { width: 640, height: 480 } } );
const report = { date: new Date().toISOString(), variants: [], methodology: 'Linear offscreen RGB luminance. Black absorbing partitions, white radiance source in right room, no direct lights. Dark-facing receiver reference is zero. One change per variant. Fixed 8x5x7 probe grid, 32px cubemap capture. Throughput: 5s, warmed, GPU-completed batches, vsync and frame limit disabled.' };
try {

	for ( const variant of variants ) {

		const page = await browser.newPage();
		const errors = [];
		page.on( 'pageerror', e => errors.push( e.message ) );
		page.on( 'console', message => {

			if ( message.type() === 'error' && ! message.text().includes( '404' ) ) errors.push( message.text() );

		} );
		await page.setRequestInterception( true );
		page.on( 'request', async request => {

			if ( request.url().includes( '/examples/jsm/tsl/lighting/LightProbeGridNode.js' ) ) {

				const original = await readFile( path.join( root, 'examples/jsm/tsl/lighting/LightProbeGridNode.js' ), 'utf8' );
				await request.respond( { status: 200, contentType: 'application/javascript', body: patch( original, variant ) } );

			} else if ( request.url().includes( '/examples/jsm/lighting/LightProbeGrid.js' ) ) {

				const original = await readFile( path.join( root, 'examples/jsm/lighting/LightProbeGrid.js' ), 'utf8' );
				await request.respond( { status: 200, contentType: 'application/javascript', body: patchBake( original, variant ) } );

			} else await request.continue();

		} );
		await page.goto( `${url}/test/benchmarks/light-probe-visibility/index.html?automated` );
		const result = await page.evaluate( async variantName => {

			const THREE = await import( '/build/three.webgpu.js' );
			const { cases, runCase } = await import( './scenes.js' );
			const adapter = await navigator.gpu.requestAdapter();
			const gpu = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description, fallback: adapter.info.isFallbackAdapter };
			if ( gpu.fallback || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( gpu ) ) || ! gpu.vendor ) throw new Error( 'Hardware GPU required' );
			const renderer = new THREE.WebGPURenderer( { antialias: false } );
			await renderer.init();
			renderer.toneMapping = THREE.NoToneMapping;
			renderer.setSize( 64, 64 );
			const measurements = [];
			for ( const config of cases ) {

				const test = await runCase( renderer, config, variantName );
				const { name, leak, lit, interior, doorway, diagnostics, distanceBytes } = test;
				measurements.push( { name, leak, lit, interior, doorway, diagnostics, distanceBytes } );
				await renderer.backend.device.queue.onSubmittedWorkDone();
				test.dispose();

			}

			// Sustained render throughput at a useful image size; no per-frame timings.
			const test = await runCase( renderer, cases.find( c => c.name === 'doorway' ), variantName );
			test.target.setSize( 512, 512 );
			for ( let i = 0; i < 40; i ++ ) test.render();
			await renderer.backend.device.queue.onSubmittedWorkDone();
			let completed = 0;
			const start = performance.now();
			while ( performance.now() - start < 5000 ) {

				for ( let i = 0; i < 8; i ++ ) test.render();
				await renderer.backend.device.queue.onSubmittedWorkDone();
				completed += 8;

			}

			const seconds = ( performance.now() - start ) / 1000;
			const throughput = { completed, seconds, fps: completed / seconds, width: 512, height: 512 };
			test.dispose();
			renderer.dispose();
			return { gpu, measurements, throughput };

		}, variant );
		for ( const measurement of result.measurements ) {

			for ( const field of [ 'leak', 'lit', 'interior' ] ) {

				const name = `${variant}-${measurement.name}-${field}.png`;
				await writeFile( path.join( output, name ), Buffer.from( measurement[ field ].png.split( ',' )[ 1 ], 'base64' ) );
				delete measurement[ field ].png;
				measurement[ field ].image = name;

			}

			if ( measurement.doorway ) measurement.doorway = measurement.interior;

		}

		const entry = { variant, ...result, errors };
		report.variants.push( entry );
		await writeFile( path.join( output, 'results.json' ), JSON.stringify( report, null, 2 ) );
		console.log( JSON.stringify( { variant, gpu: result.gpu, fps: result.throughput.fps, leaks: Object.fromEntries( result.measurements.map( m => [ m.name, m.leak.mean ] ) ), errors } ) );
		await page.close();
		if ( errors.length ) throw new Error( `${variant}: GPU errors` );

	}

} finally {

	await browser.close();
	server.close();

}

console.log( `Results: ${path.join( output, 'results.json' )}` );

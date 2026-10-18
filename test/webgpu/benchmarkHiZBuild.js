import fs from 'node:fs';
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';

// Run from the repository root. The optional baseline is an unmodified HiZNode.js.
const [ output, baseline ] = process.argv.slice( 2 );
if ( ! output ) throw new Error( 'Provide an output JSON path.' );
const server = createServer();
await new Promise( resolve => server.listen( 8082, resolve ) );
const browser = await puppeteer.launch( {
	executablePath: process.env.CHROME_PATH || ( fs.existsSync( 'C:/Program Files/Google/Chrome/Application/chrome.exe' ) ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : undefined ),
	headless: false,
	args: [ '--disable-frame-rate-limit', '--disable-gpu-vsync', '--no-sandbox' ]
} );
try {

	const page = await browser.newPage();
	await page.setRequestInterception( true );
	page.on( 'request', async request => {

		if ( request.url().endsWith( '/benchmark.html' ) ) {

			await request.respond( { status: 200, contentType: 'text/html', body: '<script type="importmap">{"imports":{"three/webgpu":"/src/Three.WebGPU.js","three/tsl":"/src/Three.TSL.js"}}</script>' } );

		} else if ( baseline && request.url().endsWith( '/HiZNode.js' ) ) {

			await request.respond( { status: 200, contentType: 'application/javascript', body: fs.readFileSync( baseline, 'utf8' ) } );

		} else {

			await request.continue();

		}

	} );
	await page.goto( 'http://localhost:8082/benchmark.html' );
	const results = await page.evaluate( async () => {

		const THREE = await import( '/src/Three.WebGPU.js' );
		const { texture } = await import( '/src/Three.TSL.js' );
		const { hiZ } = await import( '/examples/jsm/tsl/display/HiZNode.js' );
		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
		const info = adapter.info;
		const description = [ info.vendor, info.architecture, info.device, info.description ].join( ' ' );
		if ( ! description.trim() || adapter.isFallbackAdapter || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software|microsoft basic render/i.test( description ) ) throw new Error( 'Hardware required: ' + description );
		const renderer = new THREE.WebGPURenderer( { device: await adapter.requestDevice() } );
		await renderer.init();
		const device = renderer.backend.device;
		const errors = [];
		device.addEventListener( 'uncapturederror', event => errors.push( event.error.message ) );
		const results = [];
		try {

			for ( const [ width, height ] of [[ 801, 603 ], [ 1920, 1080 ], [ 3840, 2160 ]] ) {

				const data = new Float32Array( width * height );
				for ( let i = 0; i < data.length; i ++ ) data[ i ] = ( i % 997 ) / 997;
				const source = new THREE.DataTexture( data, width, height, THREE.RedFormat, THREE.FloatType );
				source.needsUpdate = true;
				const pyramid = hiZ( texture( source ) );
				const run = async duration => {

					let builds = 0;
					const start = performance.now();
					do {

						for ( let i = 0; i < 16; i ++ ) {

							pyramid.updateBefore( { renderer } );
							builds ++;

						}

						await device.queue.onSubmittedWorkDone();

					} while ( performance.now() - start < duration );

					const ms = performance.now() - start;
					return { builds, ms, buildMs: ms / builds };

				};

				await run( 1000 );
				await device.queue.onSubmittedWorkDone();
				const result = await run( 5000 );
				results.push( { width, height, adapter: description, levels: pyramid.levels.length, ...result } );
				pyramid.dispose();
				source.dispose();

			}

			if ( errors.length ) throw new Error( errors.join( '\n' ) );
			return results;

		} finally {

			renderer.dispose();

		}

	} );
	fs.writeFileSync( output, JSON.stringify( { baseline: baseline || null, results }, null, 2 ) );
	console.log( JSON.stringify( results ) );

} finally {

	await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

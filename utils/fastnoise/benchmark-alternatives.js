// Sequential, completed-work benchmark. No software GPU fallback is permitted.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import puppeteer from 'puppeteer';

const output = path.resolve( process.argv[ 2 ] || 'noise-results' );
const steps = Number( process.argv[ 3 ] || 10000 );
fs.mkdirSync( output, { recursive: true } );
const root = new URL( './', import.meta.url );
const directory = fileURLToPath( root );
const server = http.createServer( ( req, res ) => {

	const file = path.resolve( directory, '.' + new URL( req.url, 'http://localhost' ).pathname );
	if ( ! file.startsWith( path.resolve( directory ) + path.sep ) ) {

		res.writeHead( 403 ).end(); return;

	}

	try {

		res.setHeader( 'Content-Type', file.endsWith( '.js' ) ? 'text/javascript' : 'text/html' ); res.end( fs.readFileSync( file ) );

	} catch {

		res.writeHead( 404 ).end();

	}

} );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const url = `http://127.0.0.1:${server.address().port}`;
const report = { steps, node: process.version, cpu: ( await import( 'node:os' ) ).cpus()[ 0 ].model, methods: {} };
let browser;
try {

	const args = [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-frame-rate-limit', '--disable-gpu-vsync' ];
	if ( process.env.WEBGPU_BACKEND ) args.push( `--use-webgpu-adapter=${process.env.WEBGPU_BACKEND}` );
	report.browserArgs = args;
	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args, protocolTimeout: 3600000 } );
	const page = await browser.newPage();
	await page.goto( url + '/generator.html?steps=0' );
	report.gpu = await page.evaluate( async () => {

		const adapter = await navigator.gpu?.requestAdapter();
		if ( ! adapter ) return null;
		const info = adapter.info;
		return { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter };

	} );
	if ( ! report.gpu || report.gpu.isFallbackAdapter || ! /amd|nvidia|intel|apple/i.test( JSON.stringify( report.gpu ) ) || /swiftshader|llvmpipe|lavapipe/i.test( JSON.stringify( report.gpu ) ) ) throw new Error( 'Physical GPU verification failed: ' + JSON.stringify( report.gpu ) );
	console.log( 'GPU', report.gpu );
	await page.waitForFunction( () => !! window.result, { timeout: 120000 } );
	for ( const seed of [ 1, 2 ] ) {

		// Warm GPU pipelines and execution, then discard the warmup volume.
		await page.goto( `${url}/generator.html?steps=100&seed=${seed}` );
		await page.waitForFunction( () => !! window.result, { timeout: 3600000 } );
		const times = [];
		for ( let run = 0; run < 2; run ++ ) {

			const start = performance.now();
			await page.goto( `${url}/generator.html?steps=${steps}&seed=${seed}` );
			await page.waitForFunction( () => !! window.result, { timeout: 3600000, polling: 25 } );
			const elapsed = performance.now() - start;
			const data = await page.evaluate( () => ( { result: window.result, log: document.getElementById( 'log' ).textContent } ) );
			fs.writeFileSync( path.join( output, `fast-${seed}.f32` ), Buffer.from( data.result, 'base64' ) );
			times.push( elapsed ); console.log( 'FAST', seed, run, elapsed, data.log );

		}

		report.methods[ `fast-${seed}` ] = { milliseconds: times };
		await page.goto( 'about:blank' ); // final readback completed before next implementation

	}

	await browser.close(); browser = null;

} catch ( error ) {

	report.gpuError = error.message;
	console.error( error );

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

for ( const name of [ 'hierarchy', 'lattice', 'relaxation' ] ) {

	const { generate } = await import( `./alternatives/${name}.js` );
	generate( { seed: 7 } );
	for ( const seed of [ 1, 2 ] ) {

		const times = [];
		let volume;
		for ( let run = 0; run < 3; run ++ ) {

			const start = performance.now(); volume = generate( { seed } ); times.push( performance.now() - start );

		}

		if ( volume.length !== 128 * 128 * 32 || ! volume.every( x => Number.isFinite( x ) && x >= 0 && x < 1 ) ) throw new Error( `Invalid ${name} volume` );
		fs.writeFileSync( path.join( output, `${name}-${seed}.f32` ), Buffer.from( volume.buffer, volume.byteOffset, volume.byteLength ) );
		report.methods[ `${name}-${seed}` ] = { milliseconds: times };
		console.log( name, seed, times );

	}

}

fs.writeFileSync( path.join( output, 'timings.json' ), JSON.stringify( report, null, 2 ) + '\n' );

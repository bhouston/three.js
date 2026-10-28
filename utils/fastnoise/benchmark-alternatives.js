// CPU generation throughput and hardware WebGPU reference. No display-frame timings.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';

const out = path.resolve( process.argv[ 2 ] || 'noise-results' );
fs.mkdirSync( out, { recursive: true } );
const root = path.dirname( fileURLToPath( import.meta.url ) );
const cpuOnly = process.argv.includes( '--cpu-only' );
const results = cpuOnly ? JSON.parse( fs.readFileSync( path.join( out, 'timings.json' ) ) ) : { environment: { node: process.version, platform: process.platform }, runs: [] };
const save = () => fs.writeFileSync( path.join( out, 'timings.json' ), JSON.stringify( results, null, 2 ) );
const server = http.createServer( ( req, res ) => {

	const name = new URL( req.url, 'http://localhost' ).pathname;
	if ( name !== '/generator.html' ) {

		res.writeHead( 404 ).end(); return;

	}

	res.setHeader( 'Content-Type', 'text/html' );
	res.end( fs.readFileSync( path.join( root, 'generator.html' ) ) );

} );
await new Promise( resolve => server.listen( 8732, '127.0.0.1', resolve ) );
let browser;
try {

	if ( ! cpuOnly ) {

		browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-frame-rate-limit', '--disable-gpu-vsync' ], protocolTimeout: 3600000 } );
		const page = await browser.newPage();
		await page.goto( 'http://127.0.0.1:8732/generator.html?steps=100&seed=1' );
		results.environment.gpu = await page.evaluate( async () => {

			const adapter = await navigator.gpu.requestAdapter();
			if ( ! adapter ) throw new Error( 'Hardware WebGPU unavailable' );
			const info = adapter.info;
			const data = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
			if ( info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( data ) ) || ! /nvidia|amd|intel|apple|qualcomm/i.test( info.vendor ) ) throw new Error( 'Unverified hardware adapter: ' + JSON.stringify( data ) );
			return data;

		} );
		console.log( results.environment );
		await page.waitForFunction( () => !! window.result, { timeout: 3600000 } );
		for ( const seed of [ 1, 2 ] ) {

			const start = performance.now();
			await page.goto( `http://127.0.0.1:8732/generator.html?steps=10000&seed=${seed}` );
			await page.waitForFunction( () => !! window.result, { timeout: 3600000 } );
			const value = await page.evaluate( () => ( { data: window.result, log: document.querySelector( '#log' ).textContent } ) );
			const seconds = ( performance.now() - start ) / 1000;
			const name = `fast-separate-seed${seed}`;
			fs.writeFileSync( path.join( out, name + '.f32' ), Buffer.from( value.data, 'base64' ) );
			results.runs.push( { name, seed, seconds, count: 1, log: value.log, timing: 'navigation through completed readback and serialization' } );
			save(); console.log( name, seconds );
			await page.goto( 'about:blank' );

		}

		await browser.close(); browser = null;

	}

	for ( const method of [ 'filter-rank', 'hierarchy', 'local-energy' ] ) {

		const { generate } = await import( `./methods/${method}.js` );
		generate( { seed: 99 } ); // full-size warmup, excluded
		for ( const seed of [ 1, 2 ] ) {

			let count = 0, volume;
			const start = performance.now();
			do {

				volume = generate( { seed } ); count ++;

			} while ( count < 2 || performance.now() - start < 5000 );

			const seconds = ( performance.now() - start ) / 1000;
			const name = `${method}-seed${seed}`;
			results.runs = results.runs.filter( row => row.name !== name );
			fs.writeFileSync( path.join( out, name + '.f32' ), Buffer.from( volume.buffer, volume.byteOffset, volume.byteLength ) );
			results.runs.push( { name, seed, count, seconds, secondsPerVolume: seconds / count } );
			save(); console.log( name, seconds / count, count );

		}

	}

	// Seeded white control with the identical uniform volume marginal.
	const { generate } = await import( './methods/filter-rank.js' );
	for ( const seed of [ 1, 2 ] ) {

		const volume = generate( { seed, iterations: 0 } );
		fs.writeFileSync( path.join( out, `white-seed${seed}.f32` ), Buffer.from( volume.buffer ) );

	}

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

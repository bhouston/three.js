// Run from repository root: node utils/noise-comparison/benchmark-sampling.js
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import puppeteer from 'puppeteer';
import { animatedIGN, tiledBayerWeyl } from './procedural-reference.js';

const root = process.cwd();
const server = http.createServer( async ( request, response ) => {

	try {

		const file = path.resolve( root, '.' + new URL( request.url, 'http://localhost' ).pathname );
		assert.ok( file.startsWith( root + path.sep ) );
		const content = await fs.readFile( file );
		response.setHeader( 'Content-Type', { '.js': 'text/javascript', '.html': 'text/html', '.png': 'image/png' }[ path.extname( file ) ] || 'application/octet-stream' );
		response.end( content );

	} catch ( error ) {

		response.writeHead( 404 ).end( error.message );

	}

} );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
let browser;
try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath() ), headless: true, args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-frame-rate-limit', '--disable-gpu-vsync' ], protocolTimeout: 300000 } );
	const page = await browser.newPage();
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => {

		if ( /GPUValidationError|Validation Error|Error while parsing WGSL/i.test( message.text() ) ) errors.push( message.text() );

	} );
	await page.goto( `http://127.0.0.1:${server.address().port}/utils/noise-comparison/benchmark-sampling.html` );
	await page.waitForFunction( () => window.runBenchmark );
	const report = await page.evaluate( () => window.runBenchmark() );
	assert.deepEqual( errors, [] );

	for ( const row of report.rows ) {

		const reference = row.method === '3LHW' ? tiledBayerWeyl : row.method === 'Animated IGN' ? animatedIGN : null;
		if ( reference ) {

			for ( let x = 0; x < 4; x ++ ) assert.ok( Math.abs( row.readback[ x ] - reference( x, 0, row.finalIndex ) ) < 2e-5, `${row.method} readback matches CPU reference` );

		}

	}

	report.node = process.version;
	report.note = 'Three sequential trials, rotated order. Each method: 2s warmup, >=5s measurement, batches of 8 draws followed by GPU completion. 1024x1024 RGBA32F target, one scalar sample/pixel stored in R. Includes render submission, target writes and completion waits; excludes atlas loading/upload and shader compilation (warmed), accumulation, blur and display. Index cycles through 0..1023. Vsync/frame caps disabled. ms/1M is amortized sustained throughput, not per-draw GPU timestamp latency.';
	await fs.writeFile( 'utils/noise-comparison/sampling-throughput.json', JSON.stringify( report, null, 2 ) + '\n' );
	console.log( JSON.stringify( report, null, 2 ) );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

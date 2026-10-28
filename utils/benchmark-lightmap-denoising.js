/**
 * Run after `npm run build`:
 * node utils/benchmark-lightmap-denoising.js --output C:/path/to/artifacts --resolution 128 --samples 16 --reference 1024 --oidn
 * Uses a physical WebGPU adapter only. Writes JSON and a screenshot, then closes Chrome and server.
 */
import http from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const args = process.argv.slice( 2 );
const option = ( key, fallback ) => args.includes( '--' + key ) ? args[ args.indexOf( '--' + key ) + 1 ] : fallback;
const output = path.resolve( option( 'output', path.join( root, 'lightmap-denoising-results' ) ) );
const query = new URLSearchParams();
for ( const key of [ 'resolution', 'samples', 'reference', 'seconds' ] ) {

	const value = option( key, null );
	if ( value !== null ) query.set( key, value );

}

if ( args.includes( '--oidn' ) ) query.set( 'oidn', '1' );
const scene = option( 'scene', null );
if ( scene !== null ) query.set( 'benchmarkScene', scene );
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png' };
const server = http.createServer( async ( request, response ) => {

	try {

		const filename = path.resolve( root, '.' + decodeURIComponent( new URL( request.url, 'http://localhost' ).pathname ) );
		if ( ! filename.startsWith( root + path.sep ) ) throw new Error( 'Invalid path' );
		response.setHeader( 'Content-Type', mime[ path.extname( filename ) ] || 'application/octet-stream' );
		response.end( await readFile( filename ) );

	} catch {

		response.writeHead( 404 );
		response.end( 'Not found' );

	}

} );
let browser;
try {

	await mkdir( output, { recursive: true } );
	await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
	browser = await puppeteer.launch( { headless: true, protocolTimeout: 0, args: [ '--use-angle=d3d11', '--enable-unsafe-webgpu', '--disable-frame-rate-limit', '--disable-gpu-vsync' ] } );
	const page = await browser.newPage();
	await page.setViewport( { width: 1100, height: 900, deviceScaleFactor: 1 } );
	const renderingErrors = [];
	page.on( 'pageerror', error => renderingErrors.push( String( error ) ) );
	page.on( 'console', message => {

		if ( message.type() === 'error' ) {

			console.error( message.text() );
			if ( /GPUValidationError|validation error|uncaptured|device.*lost|WebGPU.*error/i.test( message.text() ) ) renderingErrors.push( message.text() );

		}

	} );
	const filename = scene === null ? 'webgpu_lightmap_denoising_benchmark.html' : 'webgpu_lightmap_baking.html';
	await page.goto( `http://127.0.0.1:${ server.address().port }/examples/${ filename }?${ query }`, { waitUntil: 'networkidle0', timeout: 180000 } );
	await page.waitForFunction( () => typeof window.runDenoisingBenchmark === 'function' || window.benchmarkError !== undefined, { timeout: 600000 } );
	const initializationError = await page.evaluate( () => window.benchmarkError );
	if ( initializationError ) throw new Error( initializationError );
	page.setDefaultTimeout( 0 );
	const result = await page.evaluate( () => window.runDenoisingBenchmark() );
	if ( renderingErrors.length > 0 ) throw new Error( 'Rendering validation failed: ' + renderingErrors.join( '\n' ) );
	await writeFile( path.join( output, 'lightmap-denoising-results.json' ), JSON.stringify( result, null, 2 ) + '\n' );
	await page.screenshot( { path: path.join( output, 'lightmap-denoising-comparison.png' ), fullPage: true } );
	console.log( JSON.stringify( result, null, 2 ) );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

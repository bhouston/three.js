// node utils/fastnoise/experiments/run-gpu.js OUTPUT_DIRECTORY PLAN_JSON
// Plan: [{name,seed,steps,identityWeight,mixedWeight}], sequential GPU-completed runs.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import puppeteer from 'puppeteer';
import { specialize } from './gpu-features.js';

const [ output, planFile ] = process.argv.slice( 2 );
const plan = JSON.parse( fs.readFileSync( planFile, 'utf8' ) );
fs.mkdirSync( output, { recursive: true } );
let html = '';
const server = http.createServer( ( req, res ) => {

	res.setHeader( 'Content-Type', 'text/html' ); res.end( html );

} );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const source = fs.readFileSync( new URL( '../generator.html', import.meta.url ), 'utf8' );
let browser;
const report = { note: 'Quality-screen elapsed times, not isolated performance benchmarks.', runs: [] };
try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--use-webgpu-adapter=d3d11', '--disable-software-rasterizer', '--disable-frame-rate-limit', '--disable-gpu-vsync' ], protocolTimeout: 3600000 } );
	const page = await browser.newPage();
	page.on( 'pageerror', error => console.error( error.message ) );
	const url = `http://127.0.0.1:${server.address().port}`;
	html = '<body>Physical GPU check</body>';
	await page.goto( url );
	report.gpu = await page.evaluate( async () => {

		const a = await navigator.gpu?.requestAdapter();
		return a ? { vendor: a.info.vendor, architecture: a.info.architecture, fallback: a.info.isFallbackAdapter ?? a.isFallbackAdapter } : null;

	} );
	if ( ! report.gpu || report.gpu.fallback !== false || ! /amd|nvidia|intel/i.test( report.gpu.vendor ) ) throw new Error( 'Physical GPU unavailable: ' + JSON.stringify( report.gpu ) );
	console.log( report.gpu );
	for ( const item of plan ) {

		if ( ! /^[a-zA-Z0-9_-]+$/.test( item.name ) ) throw new Error( 'Unsafe output name' );
		if ( ! Number.isInteger( item.steps ) || item.steps < 0 ) throw new RangeError( 'steps must be a nonnegative integer.' );
		html = specialize( source, item );
		const start = performance.now();
		await page.goto( `${url}/?mode=separate&steps=${item.steps}&seed=${item.seed || 1}` );
		await page.waitForFunction( () => !! window.result || /wgsl error/.test( document.getElementById( 'log' )?.textContent || '' ), { timeout: 1800000, polling: 100 } );
		const result = await page.evaluate( () => ( { bytes: window.result, log: document.getElementById( 'log' ).textContent } ) );
		if ( ! result.bytes ) throw new Error( result.log );
		fs.writeFileSync( path.join( output, item.name + '.f32' ), Buffer.from( result.bytes, 'base64' ) );
		report.runs.push( { ...item, milliseconds: performance.now() - start, log: result.log } );
		fs.writeFileSync( path.join( output, 'gpu-screen-' + path.basename( planFile ) ), JSON.stringify( report, null, 2 ) );
		console.log( item.name, report.runs.at( - 1 ).milliseconds );
		await page.goto( 'about:blank' );

	}

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

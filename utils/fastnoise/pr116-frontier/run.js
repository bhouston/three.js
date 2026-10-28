import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';
import puppeteer from 'puppeteer';
import { specialize } from './original/gpu-features.js';

const output = process.argv[ 2 ];
if ( ! output ) throw new Error( 'Usage: node utils/fastnoise/pr116-frontier/run.js OUTPUT_DIRECTORY [--resume]' );
const pin = 'd8b1acffaada09223b5133d965f3a7cf9028fe5b';
// Normalize Git checkout line endings before guarded source specialization.
const original = fs.readFileSync( new URL( './original/generator.html', import.meta.url ), 'utf8' ).replaceAll( '\r\n', '\n' );
const hash = value => createHash( 'sha256' ).update( value ).digest( 'hex' );
const expected = JSON.parse( fs.readFileSync( new URL( './original/volume-hashes.json', import.meta.url ) ) );
const profiles = [
	{ name: 'pr116-annealed', originalName: 'latin-anneal001-3000-s', annealTemperature: .001, annealSteps: 1500, mixedWeight: .2 },
	{ name: 'pr116-greedy', originalName: 'latin-hybrid-3000-s', mixedWeight: .2 },
	{ name: 'pr116-autocorrelation', originalName: 'latin-psd-3000-s', mixedWeight: .02, mixedKernel: 'binomial' }
].map( item => ( { steps: 3000, stratified: true, identityWeight: .25, swapFloor: 2, ...item } ) );
fs.mkdirSync( output, { recursive: true } );
const manifestFile = path.join( output, 'pr116-generation.json' );
const report = fs.existsSync( manifestFile ) ? JSON.parse( fs.readFileSync( manifestFile ) ) : { sourceCommit: pin, environment: { node: process.version, cpu: os.cpus()[ 0 ].model }, runs: [] };
report.sourceSHA256 = hash( original );
report.adaptations = [ 'Default physical backend instead of forced D3D11; no shader/objective/RNG changes.', 'Split each logical 100-iteration batch into submissions of 10 iterations, waiting for completion; retain 100-iteration swap counter, suppression update and parameter/RNG construction.', '100-step warmup per profile/seed excluded; sequential complete readbacks.' ];
const save = () => fs.writeFileSync( manifestFile, JSON.stringify( report, null, 2 ) + '\n' );
function adapted( profile ) {

	let source = specialize( original, profile );
	const before = '\tconst enc = device.createCommandEncoder();\n\tfor ( let b = 0; b < n; b ++ ) {';
	const after = '\tfor ( let start = 0; start < n; start += 10 ) {\n\tconst enc = device.createCommandEncoder();\n\tfor ( let b = start; b < Math.min( start + 10, n ); b ++ ) {';
	if ( ! source.includes( before ) ) throw new Error( 'Submission specialization guard failed' );
	source = source.replace( before, after );
	const end = '\tenc.copyBufferToBuffer( swapsBuf, 0, swapsRead, 0, 4 );\n\tdevice.queue.submit( [ enc.finish() ] );';
	if ( ! source.includes( end ) ) throw new Error( 'Readback specialization guard failed' );
	source = source.replace( end, '\tdevice.queue.submit( [ enc.finish() ] );\n\tawait device.queue.onSubmittedWorkDone();\n\t}\n\tconst enc = device.createCommandEncoder();\n' + end );
	return source;

}

let html = '<body>Physical GPU probe</body>', browser;
const server = http.createServer( ( req, res ) => {

	res.setHeader( 'Content-Type', 'text/html' ); res.end( html );

} );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-frame-rate-limit', '--disable-gpu-vsync' ], protocolTimeout: 3600000 } );
	const page = await browser.newPage();
	let failure;
	page.on( 'pageerror', error => {

		failure = error.message;

	} );
	const url = `http://127.0.0.1:${server.address().port}`;
	await page.goto( url );
	report.environment.chrome = await browser.version();
	report.environment.gpu = await page.evaluate( async () => {

		const a = await navigator.gpu?.requestAdapter();
		return a ? { vendor: a.info.vendor, architecture: a.info.architecture, fallback: a.info.isFallbackAdapter ?? a.isFallbackAdapter } : null;

	} );
	const gpu = report.environment.gpu;
	if ( ! gpu || gpu.fallback !== false || ! /amd|nvidia|intel|apple|qualcomm/i.test( gpu.vendor ) || /swiftshader|software|llvmpipe|lavapipe/i.test( JSON.stringify( gpu ) ) ) throw new Error( 'Physical GPU unavailable: ' + JSON.stringify( gpu ) );
	console.log( report.environment ); save();
	async function run( profile, seed, steps ) {

		html = adapted( profile ); failure = null;
		await page.goto( `${url}/?mode=separate&steps=${steps}&seed=${seed}` );
		const deadline = Date.now() + 1800000;
		for ( ;; ) {

			const state = await page.evaluate( () => ( { done: !! window.result, log: document.querySelector( '#log' )?.textContent } ) );
			if ( failure || /wgsl error|device lost/i.test( state.log ) ) throw new Error( failure || state.log );
			if ( state.done ) return await page.evaluate( () => ( { result: window.result, log: document.querySelector( '#log' ).textContent } ) );
			if ( Date.now() > deadline ) throw new Error( 'Generation timeout' );
			await new Promise( resolve => setTimeout( resolve, 250 ) );

		}

	}

	for ( const profile of profiles ) for ( const seed of [ 1, 2 ] ) {

		const name = `${profile.name}-seed${seed}`, file = path.join( output, name + '.f32' );
		const prior = report.runs.find( row => row.name === name );
		if ( process.argv.includes( '--resume' ) && prior && fs.existsSync( file ) && hash( fs.readFileSync( file ) ) === prior.sha256 ) continue;
		await run( profile, seed, 100 ); await page.goto( 'about:blank' );
		const start = performance.now(), value = await run( profile, seed, profile.steps );
		const secondsPerVolume = ( performance.now() - start ) / 1000, bytes = Buffer.from( value.result, 'base64' );
		if ( bytes.length !== 128 * 128 * 32 * 4 ) throw new Error( 'Unexpected volume size' );
		const sha256 = hash( bytes ), originalSHA256 = expected[ `${profile.originalName}${seed}.f32` ] ?? null;
		fs.writeFileSync( file, bytes );
		const row = { name, profile, seed, secondsPerVolume, sha256, originalSHA256, matchesOriginal: originalSHA256 === null ? null : sha256 === originalSHA256, specializedSHA256: hash( adapted( profile ) ), log: value.log, timing: 'warmup excluded; navigation through GPU-completed readback; no competing scoring work' };
		report.runs = report.runs.filter( row => row.name !== name ); report.runs.push( row ); save();
		console.log( name, secondsPerVolume, sha256, 'matches original:', row.matchesOriginal );
		await page.goto( 'about:blank' );

	}

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

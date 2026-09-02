import puppeteer from 'puppeteer';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createServer } from '../../utils/server.js';

function argument( name, fallback ) {

	const prefix = `--${ name }=`;
	const value = process.argv.find( ( entry ) => entry.startsWith( prefix ) );
	return value ? value.slice( prefix.length ) : fallback;

}

function listen( server, port ) {

	return new Promise( ( resolve, reject ) => {

		server.once( 'error', reject );
		server.listen( port, () => resolve() );

	} );

}

function closeServer( server ) {

	return new Promise( ( resolve ) => server.close( resolve ) );

}

function createStaticServer( rootDirectory ) {

	return http.createServer( ( req, res ) => {

		const pathname = decodeURIComponent( req.url.split( '?' )[ 0 ] );
		const filePath = path.join( rootDirectory, pathname );

		res.setHeader( 'Access-Control-Allow-Origin', '*' );

		if ( filePath.startsWith( rootDirectory ) === false || existsSync( filePath ) === false || statSync( filePath ).isDirectory() ) {

			res.writeHead( 404 );
			res.end( 'File not found' );
			return;

		}

		res.writeHead( 200, {
			'Content-Length': statSync( filePath ).size,
			'Content-Type': path.extname( filePath ) === '.wasm' ? 'application/wasm' : 'application/javascript'
		} );
		createReadStream( filePath ).pipe( res );

	} );

}

const models = argument( 'models', 'SmolLM2-135M-Instruct-q0f32-MLC,SmolLM2-135M-Instruct-q0f16-MLC' ).split( ',' ).filter( Boolean );
const trials = Math.max( 1, Number( argument( 'trials', '3' ) ) );
const tokens = Math.max( 2, Number( argument( 'tokens', '16' ) ) );
const context = Math.max( 1, Number( argument( 'context', '4096' ) ) );
let webllmModule = argument( 'module', '' );
const outputPath = argument( 'output', '' );
const port = Math.max( 1024, Number( argument( 'port', '1238' ) ) );
const modulePort = Math.max( 1024, Number( argument( 'module-port', String( port + 100 ) ) ) );
const timeout = Math.max( 1, Number( argument( 'timeout', String( 20 * 60 * 1000 ) ) ) );
const mode = argument( 'mode', 'completion' );
const headless = process.argv.includes( '--headless' );
const server = createServer();
let moduleServer;
let browser;

try {

	await listen( server, port );

	if ( webllmModule === '' ) {

		const webllmRoot = fileURLToPath( new URL( '../../../web-llm/', import.meta.url ) );
		const webllmIndex = path.join( webllmRoot, 'lib/index.js' );

		if ( existsSync( webllmIndex ) ) {

			moduleServer = createStaticServer( webllmRoot );
			await listen( moduleServer, modulePort );
			webllmModule = `http://localhost:${ modulePort }/lib/index.js`;

		}

	}

	browser = await puppeteer.launch( {
		headless: headless ? 'new' : false,
		args: [
			'--enable-unsafe-webgpu',
			'--ignore-gpu-blocklist',
			'--disable-gpu-driver-bug-workarounds'
		],
		defaultViewport: { width: 1000, height: 800 },
		handleSIGINT: false,
		protocolTimeout: 0
	} );

	const reports = [];

	for ( const model of models ) {

		const page = await browser.newPage();
		page.on( 'console', ( message ) => {

			if ( message.type() === 'error' || message.type() === 'warning' ) {

				console.error( `[${ model }] ${ message.type() }: ${ message.text() }` );

			}

		} );

		const url = new URL( `http://localhost:${ port }/test/benchmark/webgpu_webllm.html` );
		url.searchParams.set( 'model', model );
		url.searchParams.set( 'trials', String( trials ) );
		url.searchParams.set( 'tokens', String( tokens ) );
		url.searchParams.set( 'context', String( context ) );
		url.searchParams.set( 'mode', mode );
		if ( webllmModule ) url.searchParams.set( 'module', webllmModule );
		console.log( `Benchmarking ${ model } (${ trials } trials, ${ tokens } tokens)...` );

		await page.goto( url.href, { waitUntil: 'domcontentloaded', timeout: 120000 } );
		await page.waitForFunction( () => window.__webllmBenchmark?.done === true, { timeout } );

		const result = await page.evaluate( () => window.__webllmBenchmark );

		if ( result.error ) throw new Error( `${ model }: ${ result.error }` );
		reports.push( result.report );
		console.log( `${ model }: ${ result.report.summary.decodeTokensPerSecond.median } tok/s median decode` );
		await page.close();

	}

	let gitCommit = null;

	try {

		gitCommit = execFileSync( 'git', [ 'rev-parse', 'HEAD' ], { encoding: 'utf8' } ).trim();

	} catch ( error ) {

		// The benchmark remains useful outside a Git checkout.

	}

	const result = {
		schemaVersion: 1,
		recordedAt: new Date().toISOString(),
		gitCommit,
		reports
	};
	const json = `${ JSON.stringify( result, null, 2 ) }\n`;

	if ( outputPath ) {

		await fs.writeFile( outputPath, json );
		console.log( `Wrote ${ outputPath }` );

	} else {

		console.log( json );

	}

} finally {

	if ( browser ) await browser.close();
	if ( moduleServer ) await closeServer( moduleServer );
	await closeServer( server );

}

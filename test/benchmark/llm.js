import puppeteer from 'puppeteer';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
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

const models = argument( 'models', 'tinystories,smollm2' ).split( ',' ).filter( Boolean );
const trials = Math.max( 1, Number( argument( 'trials', '3' ) ) );
const tokens = Math.max( 2, Number( argument( 'tokens', '16' ) ) );
const outputPath = argument( 'output', '' );
const port = Math.max( 1024, Number( argument( 'port', '1236' ) ) );
const headless = process.argv.includes( '--headless' );
const server = createServer();
let browser;

try {

	await listen( server, port );
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
		await page.setCacheEnabled( false );
		page.on( 'console', ( message ) => {

			if ( message.type() === 'error' || message.type() === 'warning' ) {

				console.error( `[${ model }] ${ message.type() }: ${ message.text() }` );

			}

		} );

		const url = new URL( `http://localhost:${ port }/test/benchmark/webgpu_llm.html` );
		url.searchParams.set( 'model', model );
		url.searchParams.set( 'trials', String( trials ) );
		url.searchParams.set( 'tokens', String( tokens ) );
		console.log( `Benchmarking ${ model } (${ trials } trials, ${ tokens } tokens)...` );

		await page.goto( url.href, { waitUntil: 'domcontentloaded', timeout: 120000 } );
		await page.waitForFunction( () => window.__llmBenchmark?.done === true, { timeout: 20 * 60 * 1000 } );

		const result = await page.evaluate( () => window.__llmBenchmark );

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
	await closeServer( server );

}

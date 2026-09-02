import puppeteer from 'puppeteer';
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
		server.listen( port, resolve );

	} );

}

const iterations = Math.max( 1, Number( argument( 'iterations', '100' ) ) );
const trials = Math.max( 1, Number( argument( 'trials', '7' ) ) );
const outputPath = argument( 'output', '' );
const port = Math.max( 1024, Number( argument( 'port', '1237' ) ) );
const server = createServer();
let browser;

try {

	await listen( server, port );
	browser = await puppeteer.launch( {
		headless: false,
		args: [
			'--enable-unsafe-webgpu',
			'--ignore-gpu-blocklist',
			'--disable-gpu-driver-bug-workarounds'
		],
		defaultViewport: { width: 1000, height: 800 },
		handleSIGINT: false,
		protocolTimeout: 0
	} );

	const page = await browser.newPage();
	const url = new URL( `http://localhost:${ port }/test/benchmark/webgpu_llm_vec4.html` );
	url.searchParams.set( 'iterations', String( iterations ) );
	url.searchParams.set( 'trials', String( trials ) );

	await page.goto( url.href, { waitUntil: 'domcontentloaded', timeout: 120000 } );
	await page.waitForFunction( () => window.__vec4Benchmark?.done === true, { timeout: 20 * 60 * 1000 } );

	const result = await page.evaluate( () => window.__vec4Benchmark );

	if ( result.error ) throw new Error( result.error );

	for ( const shape of result.report.results ) {

		console.log( `${ shape.name } (${ shape.inputSize }x${ shape.outputSize })` );
		console.log( `  scalar:      ${ shape.scalar.msPerDispatch.toFixed( 4 ) } ms, ${ shape.scalar.gflops.toFixed( 1 ) } GFLOP/s` );
		console.log( `  vec4 output: ${ shape.vec4Output.msPerDispatch.toFixed( 4 ) } ms, ${ shape.vec4Output.speedup.toFixed( 2 ) }x` );
		console.log( `  vec4 dot:    ${ shape.vec4Dot.msPerDispatch.toFixed( 4 ) } ms, ${ shape.vec4Dot.speedup.toFixed( 2 ) }x` );

	}

	const json = `${ JSON.stringify( result.report, null, 2 ) }\n`;

	if ( outputPath ) {

		await fs.writeFile( outputPath, json );
		console.log( `Wrote ${ outputPath }` );

	} else {

		console.log( json );

	}

} finally {

	if ( browser ) await browser.close();
	await new Promise( ( resolve ) => server.close( resolve ) );

}

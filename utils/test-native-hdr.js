// Run from the repository root after `node utils/build/dev.js`.
// CHROME_PATH selects an existing browser. No software renderer is enabled.
import puppeteer from 'puppeteer';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createServer } from './server.js';

const experimental = process.argv.includes( '--experimental' );
const outputIndex = process.argv.indexOf( '--output' );
const output = outputIndex === - 1 ? null : path.resolve( process.argv[ outputIndex + 1 ] );
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
let browser;

try {

	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH,
		headless: true,
		args: experimental ? [ '--enable-experimental-web-platform-features' ] : []
	} );
	const results = [];
	if ( output ) await mkdir( output, { recursive: true } );
	for ( const backend of [ 'webgl', 'webgpu' ] ) {

		const page = await browser.newPage();
		await page.setViewport( { width: 1100, height: 1000 } );
		const errors = [];
		page.on( 'pageerror', error => errors.push( error.message ) );
		await page.goto( `http://127.0.0.1:${server.address().port}/examples/webgl_loader_texture_native_hdr.html?backend=${backend}` );
		await page.waitForFunction( () => window.nativeHDRTest !== undefined );
		const result = await page.evaluate( () => window.nativeHDRTest );
		results.push( { experimental, ...result, errors } );
		console.log( JSON.stringify( results[ results.length - 1 ], null, 2 ) );
		if ( output ) {

			const name = `${backend}-${experimental ? 'experimental' : 'default'}`;
			await writeFile( path.join( output, name + '.json' ), JSON.stringify( results[ results.length - 1 ], null, 2 ) );
			await page.screenshot( { path: path.join( output, name + '.png' ), fullPage: true } );

		}

		assert.equal( result.error, undefined, 'GPU initialization' );
		assert.equal( errors.length, 0, 'No uncaught browser errors' );
		assert.equal( result.images.length, 2 );
		assert.ok( ! /swiftshader|llvmpipe|lavapipe|software|basic render/i.test( result.gpu ) );
		for ( const image of result.images ) {

			assert.ok( image.defaultCanvas.every( Number.isFinite ), 'Default decoder returns finite pixels' );
			if ( experimental ) {

				assert.equal( image.error, undefined );
				assert.equal( image.passed, true, 'PQ patch values survive decode, linearization, GPU sampling and readback within 2%' );
				assert.ok( image.cpu[ 7 ] > 40, '10,000-nit patch retains HDR' );
				assert.ok( image.defaultCanvas[ 7 ] < 1.1, 'Default path loses HDR headroom' );

			} else {

				assert.match( image.error, /requires float16 Canvas2D and experimental globalHDRHeadroom/ );

			}

		}

		await page.close();

	}

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

import fs from 'fs';
import puppeteer from 'puppeteer';
const [ query, out ] = process.argv.slice( 2 );
const browser = await puppeteer.launch( { headless: 'new', args: [ '--enable-unsafe-webgpu' ], protocolTimeout: 3600000 } );
const page = await browser.newPage();
page.on( 'pageerror', e => console.log( 'pageerror:', e.message ) );
page.on( 'console', m => console.log( 'console:', m.text() ) );
await page.goto( 'http://localhost:8732/generator.html?' + query );
let last = '';
while ( true ) {

	const s = await page.evaluate( () => ( { log: document.getElementById( 'log' ).textContent, done: !! window.result } ) );
	if ( s.log !== last ) {

		process.stdout.write( s.log.slice( last.length ) ); last = s.log;

	}

	if ( s.done ) break;
	if ( /wgsl error/.test( s.log ) ) {

		await browser.close(); process.exit( 1 );

	}

	await new Promise( r => setTimeout( r, 1000 ) );

}

fs.writeFileSync( out, Buffer.from( await page.evaluate( () => window.result ), 'base64' ) );
await browser.close();

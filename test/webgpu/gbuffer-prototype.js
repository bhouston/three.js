import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import path from 'node:path';
import fs from 'node:fs';
// Run from the repository root. No software renderer flags are permitted.
const cornellVariants = [
	{ view: 'Depth' }, { view: 'Albedo' }, { view: 'Roughness' },
	{ view: 'Velocity' }, { view: 'AO' }, { view: 'Bent normals' },
	{ view: 'Hi-Z' }, { view: 'Normal' }, { view: 'SSR' },
	{ view: 'Beauty', depthPrepass: false },
	{ depthPrepass: true, hiZ: false, denoise: false },
	{ ssr: false, gtao: false }, { probes: false }, { gi: false },
	{ gi: true, visibility: false }, { visibility: true },
	{ preset: 'Low' }, { traa: false },
	{ preset: 'High', ssr: true, gtao: true, hiZ: true, denoise: true, traa: true, probes: true }
];
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const baseURL = `http://127.0.0.1:${ server.address().port }`;
const artifacts = process.env.SWARM_ARTIFACTS;
if ( artifacts ) fs.mkdirSync( artifacts, { recursive: true } );
let browser;
try {

	browser = await puppeteer.launch( { executablePath: process.env.PUPPETEER_EXECUTABLE_PATH, headless: true, args: [ '--enable-unsafe-webgpu', '--disable-gpu-sandbox' ] } );
	const report = [];
	try {

		const page = await browser.newPage(); await page.setViewport( { width: 1280, height: 800 } );
		let errors = [];
		page.on( 'console', m => {

			if ( m.type() === 'error' && ! m.text().includes( '404' ) ) errors.push( m.text() );

		} );
		page.on( 'pageerror', e => errors.push( e.stack ) );
		page.on( 'response', response => {

			if ( response.status() >= 400 && ! response.url().endsWith( '/favicon.ico' ) ) errors.push( `HTTP ${ response.status() }: ${ response.url() }` );

		} );
		for ( const scene of [ 'Cornell', 'Two rooms', 'GTAO' ] ) {

			errors = [];
			await page.goto( baseURL + '/examples/webgpu_gbuffer.html?scene=' + encodeURIComponent( scene ), { waitUntil: 'networkidle0', timeout: 60000 } );
			await page.waitForFunction( () => window.gbufferFrames > 32, { timeout: 60000 } );
			const adapter = await page.evaluate( () => window.gbufferAdapter );
			if ( ! /nvidia|amd|intel|apple|qualcomm|arm/i.test( adapter ) || /swiftshader|llvmpipe|lavapipe/i.test( adapter ) ) throw Error( 'Not hardware: ' + adapter );
			if ( artifacts ) await page.screenshot( { path: path.join( artifacts, scene.toLowerCase().replaceAll( ' ', '-' ) + '.png' ) } );
			report.push( { scene, view: 'High beauty', adapter, errors: [ ...errors ] } );
			await page.evaluate( () => window.gbufferWaitForGPU() );
			if ( errors.length ) throw Error( errors.join( '\n' ) );
			for ( const values of ( scene === 'Cornell' ? cornellVariants : [ { preset: 'Low' } ] ) ) {

				errors = [];
				const before = await page.evaluate( () => window.gbufferFrames );
				await page.evaluate( v => window.gbufferSet( v ), values );
				await page.waitForFunction( n => window.gbufferFrames > n + 12, { timeout: 60000 }, before );
				await page.evaluate( () => window.gbufferWaitForGPU() );
				if ( errors.length ) throw Error( scene + ' ' + JSON.stringify( values ) + ' ' + errors.join( '\n' ) );
				report.push( { scene, values, errors: [ ...errors ] } );
				if ( artifacts && values.view && values.view !== 'Beauty' ) await page.screenshot( { path: path.join( artifacts, 'buffer-' + values.view.toLowerCase().replaceAll( ' ', '-' ) + '.png' ) } );

			}

			await page.setViewport( { width: 1093, height: 713 } );
			const before = await page.evaluate( () => window.gbufferFrames );
			await page.waitForFunction( n => window.gbufferFrames > n + 5, { timeout: 60000 }, before );
			await page.evaluate( () => window.gbufferWaitForGPU() );
			if ( errors.length ) throw Error( 'Resize ' + errors.join( '\n' ) );

		}

		for ( const test of [ 'hi-z', 'ssr-hi-z' ] ) {

			errors = []; await page.goto( baseURL + '/test/webgpu/' + test + '.html', { waitUntil: 'networkidle0' } );
			await page.waitForFunction( () => [ 'passed', 'failed' ].includes( window.hiZTestResult?.status ), { timeout: 120000 } );
			const result = await page.evaluate( () => window.hiZTestResult ); report.push( { test, result } );
			if ( result.status !== 'passed' ) throw Error( JSON.stringify( result ) );

		}

		console.log( JSON.stringify( report, null, 2 ) );

	} finally {

		if ( artifacts ) fs.writeFileSync( path.join( artifacts, 'validation.json' ), JSON.stringify( report, null, 2 ) ); await browser.close();

	}

} finally {

	await new Promise( resolve => server.close( resolve ) );

}


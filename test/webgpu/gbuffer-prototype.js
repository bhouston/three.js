import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import path from 'node:path';
import fs from 'node:fs';
// Run from the repository root. No software renderer flags are permitted.
const cornellVariants = [
	{ view: 'Depth' }, { view: 'Albedo' }, { view: 'Roughness' }, { view: 'Velocity' },
	{ view: 'AO' }, { view: 'Bent normals' }, { view: 'Hi-Z' }, { view: 'Normal' }, { view: 'SSR' },
	{ view: 'Beauty', depthMode: 'Conservative' }, { depthMode: 'Off' },
	{ ssrTracing: 'Linear', reflectionDenoise: 'Off' },
	{ specularMethod: 'Specular probes', aoMethod: 'Off' }, { specularMethod: 'None' },
	{ diffuseMethod: 'None' }, { diffuseMethod: 'Sky IBL' },
	{ diffuseMethod: 'SSGI', view: 'SSGI' }, { view: 'Beauty' },
	{ diffuseMethod: 'Diffuse probes', visibility: false }, { visibility: true },
	{ antialiasing: 'Off' }
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
			for ( const values of ( scene === 'Cornell' ? cornellVariants : [ { depthMode: 'Conservative' }, { depthMode: 'Off' }, { diffuseMethod: 'SSGI' }, { diffuseMethod: 'Diffuse probes' } ] ) ) {

				errors = [];
				const before = await page.evaluate( () => window.gbufferFrames );
				await page.evaluate( v => window.gbufferSet( v ), values );
				await page.waitForFunction( n => window.gbufferFrames > n + 12, { timeout: 60000 }, before );
				await page.evaluate( () => window.gbufferWaitForGPU() );
				if ( errors.length ) throw Error( scene + ' ' + JSON.stringify( values ) + ' ' + errors.join( '\n' ) );
				const lighting = await page.evaluate( () => window.gbufferGetLightingState() );
				if ( lighting.skyDiffuse && lighting.probeGrids > 0 ) throw Error( 'Diffuse sky and probes are both active' );
				report.push( { scene, values, lighting, errors: [ ...errors ] } );
				if ( artifacts && values.view && values.view !== 'Beauty' ) await page.screenshot( { path: path.join( artifacts, 'buffer-' + values.view.toLowerCase().replaceAll( ' ', '-' ) + '.png' ) } );

			}

			await page.evaluate( () => window.gbufferSet( { view: 'Albedo', antialiasing: 'Off', depthMode: 'Off' } ) );
			let sampleFrame = await page.evaluate( () => window.gbufferFrames );
			await page.waitForFunction( n => window.gbufferFrames > n + 12, {}, sampleFrame );
			await page.evaluate( async () => {

				window.depthBaseline = await window.gbufferReadBuffer( 'albedo' );

			} );
			await page.evaluate( () => window.gbufferSet( { depthMode: 'Conservative' } ) );
			sampleFrame = await page.evaluate( () => window.gbufferFrames );
			await page.waitForFunction( n => window.gbufferFrames > n + 12, {}, sampleFrame );
			const changedChannels = await page.evaluate( async () => {

				const actual = await window.gbufferReadBuffer( 'albedo' );
				let changed = 0;
				for ( let i = 0; i < actual.length; i ++ ) if ( actual[ i ] !== window.depthBaseline[ i ] ) changed ++;
				return changed;

			} );
			report.push( { scene, depthPrepassChangedAlbedoChannels: changedChannels } );
			if ( changedChannels !== 0 ) throw Error( 'Pre-pass differs from disabled albedo: ' + changedChannels );
			for ( const preset of [ 'Low', 'High' ] ) {

				await page.evaluate( name => window.gbufferApplyPreset( name ), preset );
				const settings = await page.evaluate( () => window.gbufferGetSettings() );
				const visibleValues = await page.evaluate( () => window.gbufferGetControlValues() );
				for ( const [ key, value ] of Object.entries( visibleValues ) ) if ( settings[ key ] !== value ) throw Error( 'Inspector value out of sync: ' + key );
				if ( settings.depthMode !== 'Off' || settings.diffuseMethod !== 'Diffuse probes' || settings.specularMethod !== ( preset === 'High' ? 'SSR' : 'Specular probes' ) ) throw Error( 'Incorrect preset settings' );
				await page.waitForFunction( () => window.gbufferFrames > 3 );
				await page.evaluate( () => window.gbufferWaitForGPU() );
				if ( errors.length ) throw Error( errors.join( '\n' ) );

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


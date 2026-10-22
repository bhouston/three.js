import puppeteer from 'puppeteer';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createServer } from '../../../utils/server.js';
import { patch, patchBake } from './variants.js';

const output = path.join( os.homedir(), 'Coding/.benchmarks/three-light-probe-visibility' );
const report = JSON.parse( await readFile( path.join( output, 'results.json' ), 'utf8' ) );
const variants = [ ...new Set( [ ...report.variants.map( v => v.variant ), 'direct-convolution' ] ) ];
const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const url = `http://127.0.0.1:${server.address().port}`;
await mkdir( path.join( output, 'complex' ), { recursive: true } );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu' ], defaultViewport: { width: 1400, height: 900 } } );
try {

	for ( const variant of variants ) {

		const page = await browser.newPage();
		const errors = [];
		page.on( 'pageerror', e => errors.push( e.message ) );
		page.on( 'console', m => {

			if ( m.type() === 'error' && ! m.text().includes( '404' ) ) errors.push( m.text() );

		} );
		await page.setRequestInterception( true );
		page.on( 'request', async request => {

			if ( request.url().endsWith( '/examples/webgpu_lightprobes_complex.html' ) ) {

				let html = await readFile( 'examples/webgpu_lightprobes_complex.html', 'utf8' );
				let extra = '';
				if ( variant.startsWith( 'normal-' ) ) extra = `probes.normalBias = ${Number( variant.slice( 7 ) )};`;
				if ( variant.startsWith( 'view-' ) ) extra = `probes.viewBias = ${Number( variant.slice( 5 ) )};`;
				if ( variant === 'visibility-off' ) extra = 'probes.visibility = false;';
				html = html.replace( 'await probes.bake(', `${extra}
probes._testBlockers = scene.children.filter( o => o.isMesh && o.geometry.parameters.width === 0.3 && o.position.x === 0 ).map( o => ( { center: o.position.toArray(), half: [ o.geometry.parameters.width / 2, o.geometry.parameters.height / 2, o.geometry.parameters.depth / 2 ], angle: 0 } ) );
await probes.bake(` );
				html = html.replace( '// GUI', 'window.__complex = { renderer, scene, camera, probes, controls };\n// GUI' );
				await request.respond( { status: 200, contentType: 'text/html', body: html } );

			} else if ( request.url().includes( '/examples/jsm/tsl/lighting/LightProbeGridNode.js' ) ) {

				await request.respond( { status: 200, contentType: 'application/javascript', body: patch( await readFile( 'examples/jsm/tsl/lighting/LightProbeGridNode.js', 'utf8' ), variant ) } );

			} else if ( request.url().includes( '/examples/jsm/lighting/LightProbeGrid.js' ) ) {

				await request.respond( { status: 200, contentType: 'application/javascript', body: patchBake( await readFile( 'examples/jsm/lighting/LightProbeGrid.js', 'utf8' ), variant ) } );

			} else await request.continue();

		} );
		await page.goto( `${url}/examples/webgpu_lightprobes_complex.html` );
		await page.waitForFunction( () => window.__complex, { timeout: 60000 } );
		await page.evaluate( async () => {

			const adapter = await navigator.gpu.requestAdapter();
			if ( adapter.info.isFallbackAdapter || adapter.info.vendor !== 'apple' ) throw new Error( 'Hardware GPU required' );
			await new Promise( requestAnimationFrame ); await new Promise( requestAnimationFrame );
			await window.__complex.renderer.backend.device.queue.onSubmittedWorkDone();

		} );
		await page.screenshot( { path: path.join( output, 'complex', `${variant}-front.png` ) } );
		await page.evaluate( async () => {

			const c = window.__complex;
			c.camera.position.set( - 3, 2.5, 3.5 );
			c.controls.target.set( 0, 2.5, 0 );
			c.controls.update();
			await new Promise( requestAnimationFrame ); await new Promise( requestAnimationFrame );
			await c.renderer.backend.device.queue.onSubmittedWorkDone();

		} );
		await page.screenshot( { path: path.join( output, 'complex', `${variant}-door.png` ) } );
		console.log( variant, errors );
		await page.close();
		if ( errors.length ) throw new Error( `${variant}: GPU errors` );

	}

} finally {

	await browser.close(); server.close();

}

await writeFile( path.join( output, 'complex', 'variants.json' ), JSON.stringify( variants ) );

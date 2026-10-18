// Run from the repository root: node test/benchmarks/sun-sky-clouds.js [artifact-directory]
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import fs from 'node:fs/promises';
import path from 'node:path';

const output = path.resolve( process.argv[ 2 ] || 'cloud-results' );
await fs.mkdir( output, { recursive: true } );
const server = createServer();
await new Promise( resolve => server.listen( 8080, resolve ) );
let browser;
const errors = [];
try {

	browser = await puppeteer.launch( { headless: false, args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit' ], defaultViewport: { width: 1200, height: 800 } } );
	const page = await browser.newPage();
	page.on( 'pageerror', error => errors.push( error.message ) );
	await page.setRequestInterception( true );
	page.on( 'request', async request => {

		if ( request.url().endsWith( '/examples/webgpu_sky.html' ) ) {

			let html = await fs.readFile( 'examples/webgpu_sky.html', 'utf8' );
			html = html.replace( 'autoRotate: true', 'autoRotate: false' );
			html = html.replace( 'renderer.setAnimationLoop( animate );', 'window.cloudTest = { THREE, renderer, scene, camera, params, bakeSky, animate, get storageScale() { return storageScale; }, skyTarget, skyMaterial, atmosphereNode, renderPass, sun, csmSun, cloudTransmission, get bakeCount() { return bakeCount; }, get baking() { return baking; }, get environment() { return environment; } }; renderer.setAnimationLoop( animate );' );
			await request.respond( { status: 200, contentType: 'text/html', body: html } );

		} else request.continue();

	} );
	await page.goto( 'http://localhost:8080/examples/webgpu_sky.html', { timeout: 120000 } );
	await page.waitForFunction( 'window.cloudTest && cloudTest.bakeCount > 0 && !cloudTest.baking', { timeout: 120000 } );
	const adapter = await page.evaluate( () => {

		const b = window.cloudTest;
		b.renderer.setAnimationLoop( null );
		const info = b.renderer.backend.device.adapterInfo;
		if ( info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( [ info.vendor, info.description, info.architecture ].join( ' ' ) ) || ! /nvidia|amd|intel|apple|qualcomm/i.test( info.vendor ) ) throw Error( 'Hardware GPU verification blocked' );
		b.renderer.inspector.domElement.style.display = 'none';
		b.camera.lookAt( 0, 8, 0 );
		return { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };

	} );
	const states = [];
	for ( const altitude of [ 90, 30, 5, - 2 ] ) {

		for ( const coverage of [ 0, 0.3, 0.7, 1 ] ) {

			const state = await page.evaluate( async ( altitude, coverage ) => {

				const b = window.cloudTest;
				Object.assign( b.params, { altitude, cloudCoverage: coverage, cloudDensity: coverage === 1 ? 10 : 2 } );
				await b.bakeSky();
				b.renderer.render( b.scene, b.camera );
				await b.renderer.backend.device.queue.onSubmittedWorkDone();
				const pixels = await b.renderer.readRenderTargetPixelsAsync( b.skyTarget, 0, 0, b.skyTarget.width, b.skyTarget.height );
				const values = Array.from( pixels, b.THREE.DataUtils.fromHalfFloat );
				if ( ! values.every( Number.isFinite ) ) throw Error( 'Nonfinite sky radiance' );
				const t = b.cloudTransmission.value;
				if ( t < 0 || t > 1 || ( coverage === 0 && t !== 1 ) || ( coverage === 1 && altitude > 0 && t > 0.001 ) ) throw Error( 'Sun transmission endpoint failed' );
				const { atmosphere, sunLightState } = await import( 'three/addons/math/Atmosphere.js' );
				const expected = sunLightState( altitude, { ...atmosphere, aerosol: b.params.aerosol, observerAltitude: b.params.observerAltitude } ).illuminance * t * b.storageScale;
				if ( Math.abs( b.sun.intensity - expected ) > 1e-8 ) throw Error( 'Directional light does not match cloud transmittance' );
				if ( b.sun.intensity !== b.csmSun.intensity ) throw Error( 'CSM light does not match directional light' );
				return { altitude, coverage, transmission: t, sunIntensity: b.sun.intensity, readout: document.getElementById( 'readout' ).innerText };

			}, altitude, coverage );
			states.push( state );
			await page.screenshot( { path: path.join( output, `sky-${altitude}-${coverage}.jpg` ), type: 'jpeg', quality: 90 } );
			if ( altitude === 30 && ( coverage === 0 || coverage === 1 ) ) {

				const orientation = await page.evaluate( () => {

					const b = window.cloudTest;
					const orientation = b.camera.quaternion.toArray();
					b.camera.lookAt( b.camera.position.clone().add( b.sun.position.clone().normalize().multiplyScalar( 100 ) ) );
					b.renderer.render( b.scene, b.camera );
					return orientation;

				} );
				await page.screenshot( { path: path.join( output, 'sun-' + coverage + '.jpg' ), type: 'jpeg', quality: 90 } );
				await page.evaluate( orientation => window.cloudTest.camera.quaternion.fromArray( orientation ), orientation );

			}

		}

	}

	const checks = await page.evaluate( async () => {

		const b = window.cloudTest;
		Object.assign( b.params, { altitude: 30, cloudCoverage: 0, cloudDensity: 2 } );
		await b.bakeSky();
		// Compare the actual clear endpoint with the original atmosphere shader.
		b.renderPass( b.skyMaterial, b.skyTarget );
		const cloudy = await b.renderer.readRenderTargetPixelsAsync( b.skyTarget, 0, 0, 512, 256 );
		const { vec4 } = await import( 'three/tsl' );
		const clearMaterial = new b.THREE.NodeMaterial();
		clearMaterial.fragmentNode = vec4( b.atmosphereNode, 1 );
		b.renderPass( clearMaterial, b.skyTarget );
		const clear = await b.renderer.readRenderTargetPixelsAsync( b.skyTarget, 0, 0, 512, 256 );
		const differences = cloudy.reduce( ( count, value, i ) => count + ( value !== clear[ i ] ? 1 : 0 ), 0 );
		clearMaterial.dispose();
		if ( differences !== 0 ) throw Error( `Clear endpoint differs in ${differences} components` );
		Object.assign( b.params, { cloudCoverage: 1, cloudDensity: 0 } );
		await b.bakeSky();
		if ( b.cloudTransmission.value !== 1 ) throw Error( 'Zero density is not transparent' );
		const count = b.bakeCount;
		for ( let i = 0; i < 30; i ++ ) b.animate();
		await b.renderer.backend.device.queue.onSubmittedWorkDone();
		if ( b.bakeCount !== count ) throw Error( 'Static frames rebaked clouds' );
		return { clearComparedComponents: clear.length, clearDifferences: differences, zeroDensityTransmission: b.cloudTransmission.value, staticFrames: 30 };

	} );
	const throughput = await page.evaluate( async () => {

		const b = window.cloudTest;
		const results = [];
		for ( const coverage of [ 0, 0.5 ] ) {

			b.params.cloudCoverage = coverage;
			b.params.cloudDensity = 2;
			let frames = 0;
			const step = async () => {

				b.params.altitude = 20 + Math.sin( frames ) * 10;
				await b.bakeSky();
				b.renderer.render( b.scene, b.camera );
				await b.renderer.backend.device.queue.onSubmittedWorkDone();
				frames ++;

			};

			for ( let i = 0; i < 3; i ++ ) await step();
			frames = 0;
			const start = performance.now();
			do {

				await step();

			} while ( performance.now() - start < 5000 );

			const elapsed = performance.now() - start;
			results.push( { coverage, frames, elapsed, completedUpdatesPerSecond: frames * 1000 / elapsed } );

		}

		return results;

	} );
	if ( errors.length ) throw Error( errors.join( '\n' ) );
	const result = { adapter, states, checks, throughput, errors };
	await fs.writeFile( path.join( output, 'results.json' ), JSON.stringify( result, null, 2 ) );
	console.log( JSON.stringify( { adapter, checks, throughput, visualStates: states.length }, null, 2 ) );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

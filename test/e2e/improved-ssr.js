import puppeteer from 'puppeteer';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const artifacts = process.env.SWARM_ARTIFACTS || path.join( root, 'test/e2e/output-screenshots/improved-ssr' );
await mkdir( artifacts, { recursive: true } );
const { createServer } = await import( '../../utils/server.js' );
const server = createServer( { root: path.resolve( root ) } );
const browser = await puppeteer.launch( { executablePath: process.env.CHROME_EXECUTABLE || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : '/usr/bin/google-chrome' ), headless: true, args: [ '--disable-software-rasterizer', '--disable-features=AllowSwiftShaderFallback', '--enable-unsafe-webgpu' ] } );
const report = { cases: [] };
const delay = ms => new Promise( resolve => setTimeout( resolve, ms ) );
try {

	await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
	const origin = `http://127.0.0.1:${server.address().port}`;
	const page = await browser.newPage();
	await page.setViewport( { width: 960, height: 640 } );
	report.adapter = await page.goto( origin ).then( () => page.evaluate( async () => {

		const adapter = await navigator.gpu?.requestAdapter( { powerPreference: 'high-performance' } );
		if ( ! adapter ) throw new Error( 'No WebGPU adapter' );
		return { ...adapter.info.toJSON?.(), vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description, isFallbackAdapter: adapter.info.isFallbackAdapter };

	} ) );
	if ( report.adapter.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( report.adapter ) ) || ! /nvidia|intel|amd/i.test( JSON.stringify( report.adapter ) ) ) throw new Error( 'Physical WebGPU adapter unavailable' );
	for ( const scene of [ 'ssr_improved', 'ssr_improved_denoise' ] ) {

		const entry = { scene, errors: [], warnings: [] };
		report.cases.push( entry );
		const onError = error => {

			entry.errors.push( error.message ); console.log( 'ERROR', error.message.slice( 0, 1200 ) );

		};

		const onConsole = message => {

			if ( message.type() === 'error' || message.type() === 'warning' ) {

				if ( ! entry.warnings.includes( message.text() ) ) {

					entry.warnings.push( message.text() ); if ( entry.warnings.length < 8 ) console.log( message.type(), message.text().slice( 0, 1300 ) );

				}

			}

		};

		page.on( 'pageerror', onError ); page.on( 'console', onConsole );
		await page.goto( `${origin}/examples/webgpu_postprocessing_${scene}.html`, { timeout: 60000 } );
		await page.waitForFunction( () => window.ssrExperiment?.ready, { timeout: 60000 } );
		await delay( 8000 );
		entry.backend = await page.evaluate( () => ( { isWebGPU: window.ssrExperiment.renderer.backend.isWebGPUBackend, frame: window.ssrExperiment.renderer.info.render.frameCalls, triangles: window.ssrExperiment.renderer.info.render.triangles } ) );
		await page.screenshot( { path: `${artifacts}/${scene}.png` } );
		if ( ! entry.backend.isWebGPU || entry.errors.length || entry.warnings.some( x => /validation|error|invalid|failed/i.test( x ) ) ) throw new Error( `Rendering failed for ${scene}` );
		entry.tracePixels = await page.evaluate( async () => {

			const { renderer, trace } = window.ssrExperiment;
			const rt = trace.getRenderTarget();
			const data = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, rt.width, rt.height );
			let covered = 0, nonFinite = 0, rgbNonzero = 0;
			for ( let i = 0; i < data.length; i ++ ) {

				if ( ( data[ i ] & 0x7c00 ) === 0x7c00 ) nonFinite ++;
				if ( i % 4 === 3 && data[ i ] > 0 ) covered ++;
				if ( i % 4 !== 3 && data[ i ] > 0 ) rgbNonzero ++;

			}

			return { pixels: rt.width * rt.height, covered, rgbNonzero, nonFinite };

		} );
		if ( ! entry.tracePixels.covered || entry.tracePixels.nonFinite ) throw new Error( 'Trace has no valid hits or has nonfinite values' );
		for ( const mode of [ 'scene', 'environment', 'raw', 'denoised', 'coverage', 'variance', 'history', 'baseline' ] ) {

			await page.evaluate( mode => window.ssrExperiment.setMode( mode ), mode );
			await delay( 2000 );
			await page.screenshot( { path: `${artifacts}/${scene}-${mode}.png` } );

		}

		await page.evaluate( () => {

			const e = window.ssrExperiment; e.setMode( 'improved' ); e.camera.position.x += .15; e.controls.update();

		} );
		await delay( 2000 );
		await page.screenshot( { path: `${artifacts}/${scene}-motion.png` } );
		await page.setViewport( { width: 801, height: 601 } );
		await delay( 2000 );
		await page.evaluate( () => {

			const e = window.ssrExperiment; e.trace.resolutionScale = 1; e.params.resolutionScale = 1; e.denoiser.reset(); e.trace.historyValid.value = false;

		} );
		await delay( 2000 );
		await page.screenshot( { path: `${artifacts}/${scene}-full-resolution.png` } );
		entry.history = await page.evaluate( async () => {

			const e = window.ssrExperiment;
			e.renderer.setAnimationLoop( null );
			const renderFrame = () => new Promise( resolve => {

				const previousFrame = e.denoiser._frame.value;
				e.renderer.setAnimationLoop( () => {

					e.pipeline.render();
					if ( e.denoiser._frame.value === previousFrame ) return;
					e.renderer.setAnimationLoop( null );
					resolve();

				} );

			} );
			const readAge = async () => {

				const rt = e.denoiser._history[ e.denoiser._index ];
				const values = await e.renderer.readRenderTargetPixelsAsync( rt, 0, 0, rt.width, rt.height, 1 );
				let maxAge = 0, nonFinite = 0;
				for ( let i = 0; i < values.length; i ++ ) {

					const bits = values[ i ];
					if ( ( bits & 0x7c00 ) === 0x7c00 ) nonFinite ++;
					if ( i % 4 === 2 ) {

						const exponent = bits >> 10 & 31, mantissa = bits & 1023;
						const value = exponent ? ( 1 + mantissa / 1024 ) * 2 ** ( exponent - 15 ) : mantissa * 2 ** - 24;
						maxAge = Math.max( maxAge, value );

					}

				}

				return { maxAge, nonFinite };

			};

			e.denoiser.reset();
			e.trace.historyValid.value = false;
			await renderFrame();
			const reset = await readAge();
			for ( let i = 0; i < 8; i ++ ) await renderFrame();
			const accumulated = await readAge();
			e.model.position.x += 0.05;
			await renderFrame();
			const animated = await readAge();
			e.camera.rotation.y += Math.PI;
			e.camera.updateMatrixWorld();
			await renderFrame();
			const cut = await readAge();
			return { reset, accumulated, animated, cut };

		} );
		if ( entry.history.reset.maxAge !== 1 || entry.history.accumulated.maxAge <= 1 || entry.history.cut.maxAge !== 1 || Object.values( entry.history ).some( value => value.nonFinite ) ) throw new Error( `Temporal history lifecycle failed for ${scene}` );
		if ( entry.errors.length || entry.warnings.length ) throw new Error( `Debug or resize mode failed for ${scene}` );
		await page.setViewport( { width: 960, height: 640 } );
		page.off( 'pageerror', onError ); page.off( 'console', onConsole );

	}

} catch ( error ) {

	report.failure = error.stack; process.exitCode = 1;

} finally {

	await writeFile( `${artifacts}/ssr-validation.json`, JSON.stringify( report, null, 2 ) );
	await browser.close();
	if ( server.listening ) await new Promise( resolve => server.close( resolve ) );

}

console.log( JSON.stringify( report, null, 2 ).slice( 0, 5000 ) );

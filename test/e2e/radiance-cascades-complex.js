import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';

// Run against the worktree server. Rendering requires a physical WebGPU adapter.
// RC_URL and CHROME_EXECUTABLE can select a different server or browser.
const url = process.env.RC_URL || 'http://localhost:8086/examples/webgpu_radiance_cascades_complex.html?test=1';
const candidates = [ process.env.CHROME_EXECUTABLE, await puppeteer.executablePath(), 'C:/Program Files/Google/Chrome/Application/chrome.exe' ];
const executablePath = candidates.find( ( candidate ) => typeof candidate === 'string' && fs.existsSync( candidate ) );
const browser = await puppeteer.launch( { executablePath, headless: true, args: [ '--enable-gpu' ] } );
const artifacts = path.resolve( 'test/e2e/output-screenshots/radiance-cascades-complex' );
fs.mkdirSync( artifacts, { recursive: true } );

try {

	const page = await browser.newPage();
	await page.setViewport( { width: 800, height: 600 } );
	const errors = [];
	page.on( 'pageerror', ( error ) => errors.push( error.message ) );
	page.on( 'console', ( message ) => {

		if ( message.type() === 'error' && ! message.text().includes( '404' ) ) errors.push( message.text() );

	} );
	await page.goto( url, { waitUntil: 'networkidle2', timeout: 60000 } );
	await page.waitForFunction( () => window.rcDemo?.frame > 5, { timeout: 60000 } );

	const hardware = await page.evaluate( async () => {

		const device = window.rcDemo.renderer.backend.device;
		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
		const info = device.adapterInfo || adapter.info;
		return { vendor: info.vendor, architecture: info.architecture, description: info.description, fallback: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false };

	} );
	assert.equal( hardware.fallback, false );
	assert.match( JSON.stringify( hardware ), /amd|nvidia|intel|apple|qualcomm|arm/i );
	assert.doesNotMatch( JSON.stringify( hardware ), /swiftshader|llvmpipe|warp|software/i );
	await page.evaluate( () => window.rcDemo.renderer.setAnimationLoop( null ) );

	const capture = async ( name ) => {

		const data = await page.evaluate( async () => {

			const d = window.rcDemo;
			// Advance real browser frames: FRAME nodes must update between renders.
			// Warm switched pipelines before sampling their outputs.
			for ( let i = 0; i < 30; i ++ ) {

				await new Promise( requestAnimationFrame );
				d.renderPipeline.render();
				await d.renderer.backend.device.queue.onSubmittedWorkDone();

			}

			await d.renderer.backend.device.queue.onSubmittedWorkDone();
			const target = d.rc._target;
			const raw = await d.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height );
			let sum = 0, maximum = 0, nonzero = 0;
			let left = 0, right = 0;
			for ( let i = 0; i < raw.length; i += 4 ) {

				for ( let c = 0; c < 3; c ++ ) {

					const bits = raw[ i + c ];
					const exponent = ( bits >> 10 ) & 31;
					const mantissa = bits & 1023;
					const value = exponent === 0 ? mantissa * 2 ** - 24 : exponent === 31 ? Infinity : ( 1 + mantissa / 1024 ) * 2 ** ( exponent - 15 );
					if ( ! Number.isFinite( value ) || ( bits & 32768 ) ) throw new Error( 'Invalid irradiance sample.' );
					sum += value;
					maximum = Math.max( maximum, value );
					if ( value > 0 ) nonzero ++;
					if ( ( i / 4 ) % target.width < target.width / 2 ) left += value;
					else right += value;

				}

			}

			return { left, right, sum, max: maximum, nonzero, width: target.width, height: target.height, memoryBytes: d.rc.memoryBytes };

		} );
		await page.screenshot( { path: path.join( artifacts, `${ name }.png` ) } );
		return data;

	};

	const baseline = await capture( 'rc' );
	assert.ok( baseline.sum > 0 && baseline.nonzero > 10000 );
	const repeat = await capture( 'rc-repeat' );
	assert.equal( repeat.sum, baseline.sum, 'Static RC has no temporal history.' );

	const comparisons = {};
	const frameTimes = {};
	for ( const method of [ 'direct', 'probes', 'vxgi', 'rc' ] ) {

		await page.select( '#method', method );
		await capture( method );
		const pixels = PNG.sync.read( fs.readFileSync( path.join( artifacts, method + '.png' ) ) );
		let sum = 0;
		for ( let y = 200; y < 380; y ++ ) {

			for ( let x = 80; x < 710; x ++ ) {

				const i = ( y * pixels.width + x ) * 4;
				sum += pixels.data[ i ] + pixels.data[ i + 1 ] + pixels.data[ i + 2 ];

			}

		}

		comparisons[ method ] = sum / ( 630 * 180 * 3 );
		frameTimes[ method ] = await page.evaluate( async () => {

			const d = window.rcDemo;
			const samples = [];
			for ( let i = 0; i < 12; i ++ ) {

				await new Promise( requestAnimationFrame );
				const start = performance.now();
				d.renderPipeline.render();
				await d.renderer.backend.device.queue.onSubmittedWorkDone();
				samples.push( performance.now() - start );

			}

			samples.sort( ( a, b ) => a - b );
			return { medianMs: ( samples[ 5 ] + samples[ 6 ] ) / 2, minMs: samples[ 0 ], maxMs: samples[ 11 ] };

		} );

	}

	assert.ok( comparisons.rc > comparisons.direct + 1 );
	await page.click( '#warm' );
	const rightOnly = await capture( 'right-light-only' );
	assert.ok( rightOnly.sum < baseline.sum );
	assert.ok( rightOnly.right > rightOnly.left, 'The right light contributes more irradiance on the right side.' );
	await page.click( '#cool' );
	const dark = await capture( 'lights-off' );
	assert.equal( dark.sum, 0 );
	await page.click( '#warm' );
	const leftOnly = await capture( 'left-light-only' );
	assert.ok( leftOnly.left > leftOnly.right, 'The left light contributes more irradiance on the left side.' );
	await page.click( '#reset' );
	const cameras = {};
	for ( const x of [ - 4, 4 ] ) {

		await page.evaluate( ( x ) => {

			const d = window.rcDemo;
			d.camera.position.set( x, 2.5, 2.8 );
			d.controls.target.set( x, 1.8, - 2 );
			d.controls.update();

		}, x );
		cameras[ x ] = await capture( x < 0 ? 'inside-left-room' : 'inside-right-room' );
		assert.ok( cameras[ x ].sum > 0 );

	}

	await page.click( '#reset' );
	await page.setViewport( { width: 807, height: 613 } );
	await page.waitForFunction( () => window.rcDemo.camera.aspect === 807 / 613 );
	const resized = await capture( 'resized' );
	assert.equal( resized.width, 807 );
	assert.equal( resized.height, 613 );
	await page.select( '#spacing', '8' );
	const fine = await capture( 'fine' );
	assert.ok( fine.memoryBytes > resized.memoryBytes );
	await page.select( '#spacing', '16' );
	await page.setViewport( { width: 800, height: 600 } );
	await page.waitForFunction( () => window.rcDemo.camera.aspect === 800 / 600 );
	await page.click( '#reset' );
	await capture( 'final' );
	assert.deepEqual( errors, [] );
	const finalImage = PNG.sync.read( fs.readFileSync( path.join( artifacts, 'final.png' ) ) );
	fs.writeFileSync( 'examples/screenshots/webgpu_radiance_cascades_complex.jpg', jpeg.encode( finalImage, 90 ).data );
	const report = { hardware, baseline, comparisons, frameTimes, rightOnly, leftOnly, dark, cameras, resized, fine, errors };
	fs.writeFileSync( path.join( artifacts, 'results.json' ), JSON.stringify( report, null, 2 ) );
	console.log( JSON.stringify( report, null, 2 ) );

} finally {

	await browser.close();

}

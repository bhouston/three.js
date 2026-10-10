import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';

// Run against the worktree server. Rendering requires a physical WebGPU adapter.
// RC_URL and CHROME_EXECUTABLE can select a different server or browser.
const url = process.env.RC_URL || 'http://localhost:8086/examples/webgpu_radiance_cascades.html?test=1';
const candidates = [ process.env.CHROME_EXECUTABLE, await puppeteer.executablePath(), 'C:/Program Files/Google/Chrome/Application/chrome.exe' ];
const executablePath = candidates.find( ( candidate ) => typeof candidate === 'string' && fs.existsSync( candidate ) );
const browser = await puppeteer.launch( { executablePath, headless: true, args: [ '--enable-gpu' ] } );
const artifacts = path.resolve( 'test/e2e/output-screenshots/radiance-cascades' );
fs.mkdirSync( artifacts, { recursive: true } );

try {

	const page = await browser.newPage();
	await page.setViewport( { width: 800, height: 600 } );
	const errors = [];
	page.on( 'pageerror', ( error ) => errors.push( error.message ) );
	page.on( 'console', ( message ) => {

		if ( message.type() === 'error' && ! message.text().includes( '404' ) ) errors.push( message.text() );

	} );
	await page.goto( url, { waitUntil: 'domcontentloaded', timeout: 120000 } );
	await page.waitForFunction( () => window.rcDemo?.frame > 5, { timeout: 120000 } );

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

				}

			}

			return { sum, max: maximum, nonzero, width: target.width, height: target.height, memoryBytes: d.rc.memoryBytes };

		} );
		await page.screenshot( { path: path.join( artifacts, `${ name }.png` ) } );
		return data;

	};

	const baseline = await capture( 'rc' );
	assert.ok( baseline.sum > 0 && baseline.nonzero > 10000 );
	const repeat = await capture( 'rc-repeat' );
	assert.equal( repeat.sum, baseline.sum, 'Static RC has no temporal history.' );

	const comparisons = {};
	for ( const method of [ 'direct', 'probes', 'vxgi', 'rc' ] ) {

		await page.select( '#method', method );
		await capture( method );
		const pixels = PNG.sync.read( fs.readFileSync( path.join( artifacts, `${ method }.png` ) ) );
		let sum = 0;
		for ( let y = 160; y < 290; y ++ ) {

			for ( let x = 230; x < 470; x ++ ) {

				const i = ( y * pixels.width + x ) * 4;
				sum += pixels.data[ i ] + pixels.data[ i + 1 ] + pixels.data[ i + 2 ];

			}

		}

		comparisons[ method ] = sum / ( 240 * 130 * 3 );

	}

	assert.ok( comparisons.rc > comparisons.direct + 1, 'RC adds indirect light.' );

	await page.evaluate( () => {

		window.rcDemo.scene.children.find( ( object ) => object.isPointLight ).intensity = 0;

	} );
	const dark = await capture( 'light-off' );
	assert.equal( dark.sum, 0, 'No residual light remains after disabling the only source.' );
	await page.click( '#emitter' );
	const emitterBehindCamera = await page.evaluate( () => {

		const d = window.rcDemo;
		const emitter = d.scene.children.find( ( object ) => object.material?.emissiveIntensity === 30 );
		emitter.updateMatrixWorld();
		const positions = emitter.geometry.attributes.position;
		const p = d.camera.position.clone();
		let behind = true;
		for ( let i = 0; i < positions.count; i ++ ) {

			p.fromBufferAttribute( positions, i ).applyMatrix4( emitter.matrixWorld ).applyMatrix4( d.camera.matrixWorldInverse );
			behind = behind && p.z > 0;

		}

		return behind;

	} );
	assert.ok( emitterBehindCamera, 'All emitter vertices are behind the camera and absent from its depth buffer.' );
	const emissive = await capture( 'offscreen-emitter' );
	assert.ok( emissive.sum > 0, 'World-space voxel traversal includes an off-screen emitter.' );
	await page.click( '#reset' );
	const reset = await capture( 'reset' );
	const resetRelativeDifference = Math.abs( reset.sum - baseline.sum ) / baseline.sum;
	assert.ok( resetRelativeDifference < 0.001, `Revoxelization energy difference: ${ ( resetRelativeDifference * 100 ).toFixed( 3 ) }%.` );

	await page.click( '#divider' );
	const divider = await capture( 'divider' );
	assert.notEqual( divider.sum, baseline.sum );
	await page.click( '#reset' );
	await page.evaluate( () => {

		const d = window.rcDemo;
		const box = d.scene.children.find( ( object ) => object.geometry?.type === 'BoxGeometry' && object.position.y === 0.6 );
		box.position.z += 0.7;
		d.rc.volume.needsUpdate = true;

	} );
	const moved = await capture( 'moved-box' );
	assert.notEqual( moved.sum, baseline.sum );
	await page.click( '#reset' );

	await page.setViewport( { width: 807, height: 613 } );
	await page.waitForFunction( () => window.rcDemo.camera.aspect === 807 / 613 );
	const resized = await capture( 'resized' );
	assert.equal( resized.width, 807 );
	assert.equal( resized.height, 613 );
	await page.select( '#spacing', '32' );
	const coarse = await capture( 'coarse' );
	assert.ok( coarse.memoryBytes < resized.memoryBytes );
	await page.select( '#spacing', '8' );
	const fine = await capture( 'fine' );
	assert.ok( fine.memoryBytes > resized.memoryBytes );

	await page.setViewport( { width: 800, height: 600 } );
	await page.waitForFunction( () => window.rcDemo.camera.aspect === 800 / 600 );
	await page.select( '#spacing', '16' );
	await page.click( '#reset' );
	const restored = await capture( 'restored' );
	const restoredRelativeDifference = Math.abs( restored.sum - baseline.sum ) / baseline.sum;
	assert.ok( restoredRelativeDifference < 0.001, `Restored-scene energy difference: ${ ( restoredRelativeDifference * 100 ).toFixed( 3 ) }%.` );

	await page.evaluate( () => {

		const d = window.rcDemo;
		d.camera.position.set( 2, 3, 7 );
		d.controls.update();

	} );
	const orbit = await capture( 'orbit' );
	assert.ok( orbit.sum > 0 );
	await page.click( '#reset' );
	await capture( 'final' );
	assert.deepEqual( errors, [] );

	const finalImage = PNG.sync.read( fs.readFileSync( path.join( artifacts, 'final.png' ) ) );
	fs.writeFileSync( 'examples/screenshots/webgpu_radiance_cascades.jpg', jpeg.encode( finalImage, 90 ).data );
	const report = { hardware, baseline, comparisons, dark, emissive, emitterBehindCamera, resetRelativeDifference, restoredRelativeDifference, divider, moved, resized, coarse, fine, orbit, errors };
	fs.writeFileSync( path.join( artifacts, 'results.json' ), JSON.stringify( report, null, 2 ) );
	console.log( JSON.stringify( report, null, 2 ) );

} finally {

	await browser.close();

}

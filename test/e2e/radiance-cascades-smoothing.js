import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer';

// Quality/regression checks only; this harness does not time individual frames.
const baseURL = process.env.RC_BASE_URL || 'http://localhost:8088';
const executablePath = process.env.CHROME_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await puppeteer.launch( { executablePath, headless: true, args: [ '--enable-gpu' ] } );
const output = path.resolve( 'test/e2e/output-screenshots/radiance-smoothing' );
fs.mkdirSync( output, { recursive: true } );
const results = {};

try {

	for ( const name of [ 'webgpu_radiance_cascades', 'webgpu_radiance_cascades_complex' ] ) {

		const page = await browser.newPage();
		await page.setViewport( { width: 800, height: 600 } );
		const errors = [];
		page.on( 'pageerror', ( error ) => errors.push( error.message ) );
		page.on( 'console', ( message ) => {

			if ( message.type() === 'error' && ! message.text().includes( '404' ) ) errors.push( message.text() );

		} );
		await page.goto( `${ baseURL }/examples/${ name }.html?test=1`, { waitUntil: 'domcontentloaded', timeout: 120000 } );
		await page.waitForFunction( () => window.rcDemo?.frame > 5, { timeout: 120000 } );
		const hardware = await page.evaluate( async () => {

			const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
			const info = window.rcDemo.renderer.backend.device.adapterInfo || adapter.info;
			return { vendor: info.vendor, architecture: info.architecture, description: info.description, fallback: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false };

		} );
		assert.equal( hardware.fallback, false );
		assert.match( JSON.stringify( hardware ), /apple|amd|nvidia|intel|qualcomm|arm/i );
		assert.doesNotMatch( JSON.stringify( hardware ), /swiftshader|llvmpipe|lavapipe|software|warp/i );
		await page.evaluate( async () => {

			window.rcDemo.renderer.setAnimationLoop( null );
			await window.rcDemo.renderer.backend.device.queue.onSubmittedWorkDone();

		} );

		const capture = async ( label ) => {

			const data = await page.evaluate( async () => {

				const d = window.rcDemo;
				for ( let i = 0; i < 6; i ++ ) {

					await new Promise( requestAnimationFrame );
					d.renderPipeline.render();

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
						if ( ! Number.isFinite( value ) || ( bits & 32768 ) ) throw new Error( 'Invalid irradiance.' );
						sum += value;
						maximum = Math.max( maximum, value );
						if ( value > 0 ) nonzero ++;

					}

				}

				return { sum, maximum, nonzero, width: target.width, height: target.height, memoryBytes: d.rc.memoryBytes, baseDirections: d.rc.cascades[ 0 ].azimuths * d.rc.cascades[ 0 ].elevations, voxelSize: d.rc.volume.voxelSizeNode.value };

			} );
			// Hide controls for unobstructed scene comparisons, then restore them.
			await page.addStyleTag( { content: 'body > div:first-child { visibility: hidden; }' } );
			await page.screenshot( { path: path.join( output, `${ name }-${ label }.png` ) } );
			await page.evaluate( () => document.querySelector( 'head style:last-child' )?.remove() );
			return data;

		};

		const captures = {};
		for ( const preset of [ 'original', 'filtered', 'smooth', 'footprint' ] ) {

			await page.select( '#preset', preset );
			captures[ preset ] = await capture( preset );
			assert.ok( captures[ preset ].sum > 0 && captures[ preset ].nonzero > 10000 );
			await page.select( '#method', 'indirect' );
			await capture( `${ preset }-irradiance` );
			await page.select( '#method', 'rc' );
			const repeat = await capture( `${ preset }-repeat` );
			assert.equal( repeat.sum, captures[ preset ].sum, 'No temporal history.' );

		}

		await page.select( '#preset', 'smooth' );
		await page.click( '#fill' );
		await page.select( '#trace', '0' );
		captures.repaired = await capture( 'repaired' );
		assert.ok( captures.repaired.sum > 0 );
		for ( const mode of [ '1', '2' ] ) {

			await page.select( '#trace', mode );
			assert.ok( ( await capture( `repaired-${ mode }` ) ).sum > 0 );

		}

		await page.click( '#fill' );
		await page.select( '#trace', '1' );
		await page.select( '#directions', '2' );
		captures.dense = await capture( 'dense' );
		assert.ok( captures.dense.sum > 0 );
		assert.equal( captures.dense.baseDirections, 128 );
		assert.ok( captures.dense.memoryBytes > captures.smooth.memoryBytes );
		await page.select( '#directions', '1' );
		await page.select( '#resolution', '128' );
		captures.highResolution = await capture( 'high-resolution' );
		assert.ok( captures.highResolution.sum > 0 );
		assert.ok( captures.highResolution.voxelSize < captures.smooth.voxelSize );
		await page.select( '#resolution', '64' );
		await page.click( '#connect' );
		captures.connected = await capture( 'connected' );
		assert.ok( captures.connected.sum > 0 );
		for ( const method of [ 'vxgi', 'probes', 'rc' ] ) {

			await page.select( '#method', method );
			await capture( method );

		}

		await page.setViewport( { width: 807, height: 613 } );
		await page.waitForFunction( () => window.rcDemo.camera.aspect === 807 / 613 );
		await page.select( '#spacing', '8' );
		captures.resized = await capture( 'resized' );
		assert.equal( captures.resized.width, 807 );
		assert.equal( captures.resized.height, 613 );
		await page.evaluate( () => {

			window.rcDemo.scene.traverse( ( object ) => {

				if ( object.isLight ) object.intensity = 0;

			} );

		} );
		await page.click( '#fill' );
		for ( const mode of [ '0', '1', '2' ] ) {

			await page.select( '#trace', mode );
			assert.equal( ( await capture( `dark-${ mode }` ) ).sum, 0, 'No residual light.' );

		}

		if ( name === 'webgpu_radiance_cascades' ) {

			await page.click( '#emitter' );
			for ( const mode of [ '0', '1', '2' ] ) {

				await page.select( '#trace', mode );
				assert.ok( ( await capture( `offscreen-emitter-${ mode }` ) ).sum > 0 );

			}

		}

		assert.deepEqual( errors, [] );
		results[ name ] = { hardware, captures, errors };
		console.log( name, JSON.stringify( results[ name ] ) );
		await page.close();

	}

	fs.writeFileSync( path.join( output, 'results.json' ), JSON.stringify( results, null, 2 ) );

} finally {

	await browser.close();

}

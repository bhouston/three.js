// Hardware browser checks, with test access injected only into the served HTML.
// Usage: node check-noise-example.js --root REPO --artifacts OUTPUT
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { createRequire } from 'node:module';

const args = process.argv.slice( 2 );
const option = ( name, fallback ) => args.includes( name ) ? args[ args.indexOf( name ) + 1 ] : fallback;
const root = path.resolve( option( '--root', '.' ) );
const artifacts = path.resolve( option( '--artifacts', 'noise-example-check' ) );
const require = createRequire( path.join( root, 'package.json' ) );
const puppeteer = require( 'puppeteer' );
const { PNG } = require( 'pngjs' );
const manifest = JSON.parse( await fs.readFile( path.join( root, 'examples/textures/noises/experimental/manifest.json' ), 'utf8' ) );
const requests = new Map();
let failURL, delayURL;
const server = http.createServer( async ( request, response ) => {

	try {

		const url = new URL( request.url, 'http://localhost' ).pathname;
		requests.set( url, ( requests.get( url ) || 0 ) + 1 );
		if ( url === failURL ) {

			failURL = null;
			response.writeHead( 503 ).end( 'Deliberate retry test' );
			return;

		}

		if ( url === delayURL ) await new Promise( resolve => setTimeout( resolve, 600 ) );
		if ( url === '/build/three.webgpu.js' || url === '/build/three.tsl.js' ) {

			response.setHeader( 'Content-Type', 'text/javascript' );
			response.end( `export * from '/src/${ url.includes( 'webgpu' ) ? 'Three.WebGPU.js' : 'Three.TSL.js' }';` );
			return;

		}

		const file = path.resolve( root, '.' + url );
		if ( ! file.startsWith( root + path.sep ) ) throw new Error( 'Invalid path' );
		let content = await fs.readFile( file );
		if ( url.endsWith( '/webgpu_tsl_blue_noise.html' ) ) {

			content = content.toString().replace( '</script>\n\n\t</body>', 'window.__noiseTest = { get renderer() { return renderer; }, params, get samples() { return samples; }, index, animate, get targets() { return targets; } };\n\t\t</script>\n\n\t</body>' );
			// Accept either checkout newline convention.
			if ( ! content.includes( 'window.__noiseTest' ) ) content = content.replace( /<\/script>\s*<\/body>/, 'window.__noiseTest = { get renderer() { return renderer; }, params, get samples() { return samples; }, index, animate, get targets() { return targets; } };\n</script>\n</body>' );

		}

		const mime = { '.js': 'text/javascript', '.html': 'text/html', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };
		response.setHeader( 'Content-Type', mime[ path.extname( file ) ] || 'application/octet-stream' );
		response.end( content );

	} catch ( error ) {

		response.writeHead( 404 ).end( error.message );

	}

} );
await fs.mkdir( artifacts, { recursive: true } );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
let browser;
const report = { options: [], errors: [], assertions: [] };

try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer' ], protocolTimeout: 120000 } );
	const page = await browser.newPage();
	await page.setViewport( { width: 1000, height: 700, deviceScaleFactor: 1 } );
	page.on( 'pageerror', error => report.errors.push( error.message ) );
	page.on( 'console', message => {

		if ( /GPUValidationError|Validation Error|Error while parsing WGSL/i.test( message.text() ) ) report.errors.push( message.text() );

	} );
	await page.goto( `http://127.0.0.1:${ server.address().port }/examples/webgpu_tsl_blue_noise.html` );
	await page.waitForFunction( () => window.__noiseTest?.targets && document.querySelectorAll( 'select' )[ 1 ]?.options.length === 24, { timeout: 60000 } );
	report.gpu = await page.evaluate( async () => {

		const test = window.__noiseTest;
		await test.renderer.init();
		test.renderer.setAnimationLoop( null );
		test.params.animate = false;
		test.params.temporalBlur = false;
		const adapter = await navigator.gpu.requestAdapter();
		const info = adapter?.info;
		if ( ! info || info.isFallbackAdapter || ! /nvidia|amd|intel|apple|qualcomm/i.test( info.vendor ) || /swiftshader|software|llvmpipe|lavapipe/i.test( JSON.stringify( info ) ) ) throw new Error( 'Physical GPU verification blocked' );
		if ( ! test.renderer.backend.isWebGPUBackend ) throw new Error( 'Example did not use WebGPU' );
		return { vendor: info.vendor, architecture: info.architecture, description: info.description, backend: 'WebGPU' };

	} );
	const selectNoise = async label => {

		await page.evaluate( value => {

			const select = document.querySelectorAll( 'select' )[ 1 ];
			select.value = value;
			select.dispatchEvent( new Event( 'change', { bubbles: true } ) );

		}, label );

	};

	const waitNoise = label => page.waitForFunction( value => document.getElementById( 'blueLabel' ).textContent === value && ! document.getElementById( 'noiseStatus' ).textContent.startsWith( 'Loading' ), {}, label );
	const atlasURL = entry => `/examples/textures/noises/experimental/${ entry.file }`;
	failURL = atlasURL( manifest.entries[ 0 ] );
	await selectNoise( manifest.entries[ 0 ].label );
	await page.waitForFunction( () => document.getElementById( 'noiseStatus' ).textContent.startsWith( 'Could not load' ) );
	assert.equal( await page.evaluate( () => window.__noiseTest.params.noise ), 'FAST Spatio-Temporal' );
	await selectNoise( manifest.entries[ 0 ].label );
	await waitNoise( manifest.entries[ 0 ].label );
	report.assertions.push( 'Failed request keeps previous selection and retries successfully' );
	delayURL = atlasURL( manifest.entries[ 1 ] );
	await selectNoise( manifest.entries[ 1 ].label );
	await page.waitForFunction( () => document.getElementById( 'noiseStatus' ).textContent.startsWith( 'Loading' ) );
	await selectNoise( 'White' );
	await waitNoise( 'White' );
	await new Promise( resolve => setTimeout( resolve, 800 ) );
	assert.equal( await page.$eval( '#blueLabel', element => element.textContent ), 'White' );
	delayURL = null;
	report.assertions.push( 'Late atlas completion cannot overwrite a newer selection' );
	const labels = await page.evaluate( () => Array.from( document.querySelectorAll( 'select' )[ 1 ].options, option => option.value ) );
	assert.equal( labels.length, manifest.entries.length + 4 );
	for ( const label of labels ) {

		await selectNoise( label );
		await waitNoise( label );
		const result = { label };
		for ( const mode of [ 'Sphere', 'Uniform' ] ) {

			await page.evaluate( mode => {

				const select = document.querySelectorAll( 'select' )[ 0 ];
				select.value = mode;
				select.dispatchEvent( new Event( 'change', { bubbles: true } ) );

			}, mode );
			await new Promise( resolve => setTimeout( resolve, 20 ) );
			result[ mode ] = await page.evaluate( async () => {

				const test = window.__noiseTest;
				test.animate();
				await test.renderer.backend.device.queue.onSubmittedWorkDone();
				const target = test.targets[ 1 ];
				const x = Math.floor( target.width * 0.75 ), y = Math.floor( target.height * 0.5 );
				const data = await test.renderer.readRenderTargetPixelsAsync( target, x, y, 1, 1 );
				return { x, y, value: data[ 0 ], index: test.index.value, samples: test.samples };

			} );
			assert.ok( Number.isFinite( result[ mode ].value ) && result[ mode ].value >= 0 && result[ mode ].value <= 1, `${ label } ${ mode } has valid readback` );

		}

		const entry = manifest.entries.find( entry => entry.label === label );
		const atlasPath = entry ? `experimental/${ entry.file }` : label === 'FAST Spatio-Temporal' ? 'fast/gauss1-ema01-spatio-temporal.png' : label !== 'White' ? 'fast/gauss1-ema01-spatial.png' : null;
		if ( atlasPath ) {

			const png = PNG.sync.read( await fs.readFile( path.join( root, 'examples/textures/noises', atlasPath ) ) );
			const { x, y, value } = result.Uniform;
			const expected = ( png.data[ ( ( y % 128 ) * png.width + x % 128 ) * 4 ] + 0.5 ) / 256;
			assert.ok( Math.abs( value - expected ) < 1e-6, `${ label }: readback ${ value }, expected ${ expected }` );
			result.expectedMidpoint = expected;
			if ( entry ) {

				result.playbackFrames = [];
				for ( const frame of [ 1, 31, 32, 1024 ] ) {

					const actual = await page.evaluate( async ( frame ) => {

						const test = window.__noiseTest;
						test.index.value = frame;
						test.animate();
						await test.renderer.backend.device.queue.onSubmittedWorkDone();
						const target = test.targets[ 1 ];
						return ( await test.renderer.readRenderTargetPixelsAsync( target, Math.floor( target.width * 0.75 ), Math.floor( target.height * 0.5 ), 1, 1 ) )[ 0 ];

					}, frame );
					const cycle = Math.floor( frame / 32 ) % 128;
					const px = ( x + 37 * cycle ) % 128, py = frame % 32 * 128 + ( y + 71 * cycle ) % 128;
					const midpoint = ( png.data[ ( py * png.width + px ) * 4 ] + 0.5 ) / 256;
					assert.ok( Math.abs( actual - midpoint ) < 1e-6, `${ label } frame ${ frame }: ${ actual } != ${ midpoint }` );
					result.playbackFrames.push( frame );

				}

			}

		}

		report.options.push( result );

	}

	await selectNoise( manifest.entries[ 0 ].label );
	await waitNoise( manifest.entries[ 0 ].label );
	assert.equal( requests.get( atlasURL( manifest.entries[ 0 ] ) ), 2 );
	assert.equal( requests.get( '/examples/textures/noises/experimental/manifest.json' ), 1 );
	report.assertions.push( 'Successful atlas cache prevents repeat downloads; catalog fetched once' );
	await page.evaluate( () => window.__noiseTest.animate() );
	await page.screenshot( { path: path.join( artifacts, 'noise-example-dropdown.png' ) } );
	assert.deepEqual( report.errors, [] );
	report.passed = true;

} catch ( error ) {

	report.passed = false;
	report.failure = error.stack;
	process.exitCode = 1;

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );
	await fs.writeFile( path.join( artifacts, 'noise-example-check.json' ), JSON.stringify( report, null, 2 ) + '\n' );
	console.log( JSON.stringify( report, null, 2 ) );

}

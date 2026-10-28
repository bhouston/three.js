// Hardware integration check. Test access is injected into served HTML only.
// node utils/noise-comparison/check-example.js --root . --artifacts OUTPUT [--backend webgl]
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { createRequire } from 'node:module';
import { animatedIGN, tiledBayerWeyl } from './procedural-reference.js';

const args = process.argv.slice( 2 );
const option = ( key, fallback ) => args.includes( key ) ? args[ args.indexOf( key ) + 1 ] : fallback;
const backend = option( '--backend', 'webgpu' );
assert.ok( [ 'webgpu', 'webgl' ].includes( backend ) );
const root = path.resolve( option( '--root', '.' ) );
const artifacts = path.resolve( option( '--artifacts', 'noise-example-check' ) );
const require = createRequire( path.join( root, 'package.json' ) );
const puppeteer = require( 'puppeteer' );
const requests = [];
let failURL, delayURL;
const hook = 'window.__noiseTest = { get renderer() { return renderer; }, params, get ready() { return computeQuad !== undefined; }, get samples() { return samples; }, index, animate, get targets() { return targets; }, async complete() { if (renderer.backend.isWebGPUBackend) await renderer.backend.device.queue.onSubmittedWorkDone(); else renderer.backend.gl.finish(); } };';
const server = http.createServer( async ( request, response ) => {

	try {

		const url = new URL( request.url, 'http://localhost' ).pathname;
		const record = { url, bytes: 0, started: performance.now() };
		requests.push( record );
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
		if ( url.endsWith( '/webgpu_tsl_blue_noise.html' ) ) content = content.toString().replace( 'new THREE.WebGPURenderer()', backend === 'webgl' ? 'new THREE.WebGPURenderer( { forceWebGL: true } )' : 'new THREE.WebGPURenderer()' ).replace( /<\/script>\s*<\/body>/, hook + '\n</script>\n</body>' );
		const mime = { '.js': 'text/javascript', '.html': 'text/html', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };
		response.setHeader( 'Content-Type', mime[ path.extname( file ) ] || 'application/octet-stream' );
		record.bytes = Buffer.byteLength( content );
		record.elapsedMs = performance.now() - record.started;
		response.end( content );

	} catch ( error ) {

		response.writeHead( 404 ).end( error.message );

	}

} );
await fs.mkdir( artifacts, { recursive: true } );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
let browser;
const report = { backend, options: [], errors: [], assertions: [], startup: [], timingNote: 'Local HTTP selection-to-GPU-completion includes decode/upload and shader work; it is not an Internet download time or intrinsic generator cost. Arithmetic methods have zero mask bytes and zero precomputation.' };

try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-frame-rate-limit', '--disable-gpu-vsync' ], protocolTimeout: 120000 } );
	const page = await browser.newPage();
	await page.setViewport( { width: 1000, height: 700, deviceScaleFactor: 1 } );
	page.on( 'pageerror', error => report.errors.push( error.message ) );
	page.on( 'console', message => {

		if ( /GPUValidationError|Validation Error|Error while parsing WGSL/i.test( message.text() ) ) report.errors.push( message.text() );

	} );
	const navigationStart = performance.now();
	await page.goto( `http://127.0.0.1:${ server.address().port }/examples/webgpu_tsl_blue_noise.html` );
	await page.waitForFunction( () => window.__noiseTest?.ready && document.querySelectorAll( 'select' )[ 1 ]?.options.length === 6, { timeout: 60000 } );
	report.gpu = await page.evaluate( async () => {

		const test = window.__noiseTest;
		await test.renderer.init();
		test.renderer.setAnimationLoop( null );
		test.params.animate = false;
		test.params.temporalBlur = false;
		await test.complete();
		if ( ! test.renderer.backend.isWebGPUBackend ) {

			const gl = test.renderer.backend.gl;
			const debug = gl.getExtension( 'WEBGL_debug_renderer_info' );
			const description = debug ? gl.getParameter( debug.UNMASKED_RENDERER_WEBGL ) : '';
			if ( ! /nvidia|amd|intel|apple|qualcomm/i.test( description ) || /swiftshader|software|llvmpipe|lavapipe/i.test( description ) ) throw new Error( 'Physical WebGL GPU verification blocked: ' + description );
			return { description, backend: 'WebGL' };

		}

		const adapter = await navigator.gpu.requestAdapter();
		const info = adapter?.info;
		if ( ! info || info.isFallbackAdapter || ! /nvidia|amd|intel|apple|qualcomm/i.test( info.vendor ) || /swiftshader|software|llvmpipe|lavapipe/i.test( JSON.stringify( info ) ) ) throw new Error( 'Physical GPU verification blocked' );
		if ( ! test.renderer.backend.isWebGPUBackend ) throw new Error( 'Example did not use WebGPU' );
		return { vendor: info.vendor, architecture: info.architecture, description: info.description, backend: 'WebGPU' };

	} );
	report.initialPageStartupMs = performance.now() - navigationStart;
	assert.equal( requests.filter( entry => entry.url.endsWith( '.png' ) ).length, 0, 'Default method must not request PNGs' );
	report.assertions.push( 'Default method initializes without requesting an image' );
	const labels = await page.evaluate( () => Array.from( document.querySelectorAll( 'select' )[ 1 ].options, option => option.value ) );
	const select = async ( label, control = 1 ) => {

		await page.evaluate( ( value, control ) => {

			const select = document.querySelectorAll( 'select' )[ control ];
			select.value = value;
			select.dispatchEvent( new Event( 'change', { bubbles: true } ) );

		}, label, control );

	};

	const waitNoise = label => page.waitForFunction( label => document.getElementById( 'blueLabel' ).textContent.includes( label ) && ! document.getElementById( 'noiseStatus' )?.textContent.startsWith( 'Loading' ), { timeout: 60000 }, label );
	const render = async frame => page.evaluate( async frame => {

		const test = window.__noiseTest;
		test.index.value = frame;
		test.animate();
		await test.complete();
		const target = test.targets[ 1 ];
		const points = [[ Math.floor( target.width * .75 ), 30 ], [ Math.floor( target.width * .75 ) + 17, 117 ]];
		const values = [];
		for ( const [ x, y ] of points ) values.push( { x, y, value: ( await test.renderer.readRenderTargetPixelsAsync( target, x, test.renderer.backend.isWebGPUBackend ? y : target.height - 1 - y, 1, 1 ) )[ 0 ] } );
		return values;

	}, frame );
	// Check arithmetic methods first, before any FAST texture has been selected.
	const arithmetic = labels.filter( label => /3LHW|IGN|White/i.test( label ) );
	for ( const label of [ ...arithmetic, ...labels.filter( label => ! arithmetic.includes( label ) ) ] ) {

		const start = performance.now(), requestStart = requests.length;
		await select( label );
		await waitNoise( label );
		await render( 0 );
		report.startup.push( { label, selectionToGPUCompletedMs: performance.now() - start, pngURLs: requests.slice( requestStart ).filter( entry => entry.url.endsWith( '.png' ) ).map( entry => entry.url ), pngBytes: requests.slice( requestStart ).filter( entry => entry.url.endsWith( '.png' ) ).reduce( ( sum, entry ) => sum + entry.bytes, 0 ), intrinsicPrecomputeMs: arithmetic.includes( label ) ? 0 : null } );
		const result = { label, modes: [] };
		for ( const mode of [ 'Sphere', 'Uniform' ] ) {

			await select( mode, 0 );
			await page.waitForFunction( value => window.__noiseTest.params.mode === value, {}, mode );
			const values = await render( 0 );
			for ( const point of values ) assert.ok( Number.isFinite( point.value ) && point.value >= 0 && point.value <= 1, `${ label } ${ mode } valid readback` );
			result.modes.push( mode );

		}

		if ( /3LHW|IGN/.test( label ) ) {

			result.frames = [];
			result.maximumReferenceError = 0;
			for ( const frame of [ 0, 1, 31, 32, 128, 1024 ] ) {

				const points = await render( frame );
				for ( const point of points ) {

					const expected = /3LHW/.test( label ) ? tiledBayerWeyl( point.x, point.y, frame ) : animatedIGN( point.x, point.y, frame );
					const tolerance = /IGN/.test( label ) ? 2e-5 : 1e-7;
					const difference = Math.abs( point.value - expected );
					result.maximumReferenceError = Math.max( result.maximumReferenceError, difference );
					assert.ok( difference < tolerance, `${ label } frame ${ frame } (${ point.x },${ point.y }): ${ point.value } != ${ expected }` );

				}

				result.frames.push( { frame, points } );

			}

		}

		report.options.push( result );
		if ( arithmetic.includes( label ) ) assert.equal( requests.filter( entry => entry.url.endsWith( '.png' ) ).length, 0, 'Procedural and white choices must not request PNGs' );

	}

	report.assertions.push( 'All six choices render in Sphere and Uniform modes', 'Procedural readbacks match independent CPU formulas at six frame indices' );
	const before = requests.filter( entry => entry.url.endsWith( '.png' ) ).length;
	for ( const label of labels ) {

		await select( label );
		await waitNoise( label );

	}

	assert.equal( requests.filter( entry => entry.url.endsWith( '.png' ) ).length, before, 'Loaded atlases are cached' );
	report.assertions.push( 'Reselecting methods reuses loaded atlases' );
	await select( arithmetic.find( label => /3LHW/.test( label ) ) );
	await waitNoise( arithmetic.find( label => /3LHW/.test( label ) ) );
	for ( const mode of [ 'Sphere', 'Uniform' ] ) {

		await select( mode, 0 );
		await page.waitForFunction( value => window.__noiseTest.params.mode === value, {}, mode );
		await render( 0 );
		await page.screenshot( { path: path.join( artifacts, `procedural-noise-${ mode.toLowerCase() }.png` ) } );

	}

	// Fresh document gives an empty loader cache for retry and race checks.
	await page.reload();
	await page.waitForFunction( () => window.__noiseTest?.ready && document.querySelectorAll( 'select' )[ 1 ]?.options.length === 6, { timeout: 60000 } );
	await page.evaluate( async () => {

		window.__noiseTest.renderer.setAnimationLoop( null );
		await window.__noiseTest.renderer.init();
		await window.__noiseTest.complete();

	} );
	const atlasMethods = report.startup.filter( entry => entry.pngURLs.length > 0 );
	assert.equal( atlasMethods.length, 2 );
	failURL = atlasMethods[ 0 ].pngURLs[ 0 ];
	await select( atlasMethods[ 0 ].label );
	await page.waitForFunction( () => document.getElementById( 'noiseStatus' ).textContent.startsWith( 'Could not load' ) );
	assert.equal( await page.evaluate( () => window.__noiseTest.params.noise ), '3LHW' );
	await select( atlasMethods[ 0 ].label );
	await waitNoise( atlasMethods[ 0 ].label );
	report.assertions.push( 'A failed atlas request retains the active procedural selection and retries successfully' );
	delayURL = atlasMethods[ 1 ].pngURLs[ 0 ];
	await select( atlasMethods[ 1 ].label );
	await page.waitForFunction( () => document.getElementById( 'noiseStatus' ).textContent.startsWith( 'Loading' ) );
	await select( 'White' );
	await waitNoise( 'White' );
	await new Promise( resolve => setTimeout( resolve, 800 ) );
	assert.equal( await page.$eval( '#blueLabel', element => element.textContent ), 'White' );
	report.assertions.push( 'A late atlas completion cannot overwrite a newer arithmetic selection' );
	assert.deepEqual( report.errors, [] );
	report.pngRequests = requests.filter( entry => entry.url.endsWith( '.png' ) );
	report.passed = true;

} catch ( error ) {

	report.passed = false;
	report.failure = error.stack;
	process.exitCode = 1;

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );
	await fs.writeFile( path.join( artifacts, 'procedural-example-check.json' ), JSON.stringify( report, null, 2 ) + '\n' );
	console.log( JSON.stringify( report, null, 2 ) );

}

/* global window, document, navigator */
// Compare the SSR-only branch with the merged SSGI and temporal-AA audits.
// Scene recovery targets are converged outputs of each method, not physical ground truth.
import puppeteer from 'puppeteer';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import { URL } from 'node:url';
import { createServer } from '../../utils/server.js';

const directory = process.env.MERGE_VALIDATION_OUTPUT || '/tmp/three-interior-merge-validation';
const baseline = process.env.MERGE_VALIDATION_BASE || '876361339e40';
const methods = ( process.env.MERGE_VALIDATION_METHODS || 'before,after' ).split( ',' );
if ( methods.some( method => ! [ 'before', 'after' ].includes( method ) ) ) throw new Error( 'Unsupported method' );
const files = [ 'SSGINode.js', 'CoverageDenoiseNode.js', 'TRAANode.js', 'TemporalReprojectNode.js' ];
const original = Object.fromEntries( files.map( file => {

	const path = 'examples/jsm/tsl/display/' + file;
	return [ '/' + path, execFileSync( 'git-dedup', [ 'show', `${ baseline }:${ path }` ], { encoding: 'utf8' } ) ];

} ) );
// TAAUtils only adds a pure helper export. Retain it so current test modules can import it;
// the baseline pipeline never calls that helper.
const source = await readFile( 'examples/webgpu_interior.html', 'utf8' );
const coverageTests = ( await readFile( 'test/unit/addons/tsl/CoverageDenoiseNode.tests.js', 'utf8' ) ).replace( "QUnit.test( 'disocclusion restarts per-pixel history age independently of coverage'", "QUnit.skip( 'disocclusion restarts per-pixel history age independently of coverage'" );
await mkdir( directory, { recursive: true } );
const server = createServer();
let browser;
const results = { baseline, units: [], scenes: [], notes: [ 'Independent identical bake settings per method; no physical image reference.', 'Move-stop errors compare against each method\'s own converged image, including lighting feedback.', 'Current test definitions run against both implementations; baseline failures are expected. The age-attachment instrumentation test is skipped on baseline, which lacks that attachment; an output-only disocclusion test runs on both.' ] };
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	const origin = `http://localhost:${ server.address().port }`;
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-background-timer-throttling', '--disable-renderer-backgrounding' ],
		defaultViewport: { width: 800, height: 600 }
	} );
	const hardwarePage = await browser.newPage();
	await hardwarePage.goto( origin );
	results.hardware = await hardwarePage.evaluate( async () => {

		const adapter = await navigator.gpu.requestAdapter();
		const info = adapter.info;
		const gl = document.createElement( 'canvas' ).getContext( 'webgl2' );
		const debug = gl.getExtension( 'WEBGL_debug_renderer_info' );
		return { vendor: info.vendor, architecture: info.architecture, isFallbackAdapter: info.isFallbackAdapter, webglRenderer: gl.getParameter( debug.UNMASKED_RENDERER_WEBGL ) };

	} );
	if ( ! results.hardware.vendor || results.hardware.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( results.hardware ) ) ) throw new Error( 'Hardware GPU required' );
	await hardwarePage.close();
	for ( const method of methods ) {

		const page = await browser.newPage();
		await page.setRequestInterception( true );
		let html = null;
		page.on( 'request', request => {

			const path = new URL( request.url() ).pathname;
			if ( path.endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
			if ( html && path === '/examples/webgpu_interior.html' ) return request.respond( { status: 200, contentType: 'text/html', body: html } );
			if ( method === 'before' && path.endsWith( '/CoverageDenoiseNode.tests.js' ) ) return request.respond( { status: 200, contentType: 'text/javascript', body: coverageTests } );
			if ( method === 'before' && original[ path ] ) return request.respond( { status: 200, contentType: 'text/javascript', body: original[ path ] } );
			return request.continue();

		} );
		if ( process.env.MERGE_VALIDATION_SKIP_UNITS !== '1' ) for ( const filter of [ 'SSGINode', 'CoverageDenoiseNode', 'TemporalPipeline' ] ) {

			await page.goto( `${ origin }/test/unit/UnitTestsAddons.html?filter=${ filter }` );
			await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 120000 } );
			const test = await page.evaluate( () => ( { stats: window._QUnitStats, failures: Array.from( document.querySelectorAll( '#qunit-tests > .fail' ), node => node.textContent ), skipped: document.getElementById( 'qunit-tests' ).textContent.includes( 'SKIPPED' ) } ) );
			results.units.push( { method, filter, ...test } );
			console.log( method, filter, test.stats );
			if ( method === 'after' && ( test.stats.failed || test.skipped || ! test.stats.passed ) ) throw new Error( 'Merged tests failed' );

		}
		if ( process.env.MERGE_VALIDATION_UNIT_ONLY !== '1' ) for ( const model of [ 'Cornell Box (Mirror Wall)', 'Primitives Floor (Metallic Mirror)' ] ) {

			html = source.replace( "model: 'Dual Rooms'", `model: ${ JSON.stringify( model ) }` )
				.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
				.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
				.replace( 'renderer.setAnimationLoop( animate );', 'window.__merge = { renderer, camera, controls, params, scene, prePass, scenePass, traaPass, renderPipeline, animate, updateOutput, resetScreenSpaceLighting }; renderer.setAnimationLoop( animate );' );
			const errors = [];
			const onError = error => errors.push( error.message );
			const onConsole = message => { if ( message.type() === 'error' ) errors.push( message.text() ); };
			page.on( 'pageerror', onError ); page.on( 'console', onConsole );
			await page.goto( `${ origin }/examples/webgpu_interior.html` );
			await page.waitForFunction( () => window.__merge && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 240000 } );
			await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
			const slug = model.startsWith( 'Cornell' ) ? 'cornell' : 'floor';
			const diagnostic = await page.evaluate( async () => {

				const s = window.__merge;
				s.renderer.setAnimationLoop( null ); s.renderer._animation.stop();
				s.controls.enableDamping = false;
				window.__step = () => { s.renderer.info.reset(); s.renderer._nodes.nodeFrame.update(); s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId; s.animate(); };
				window.__drain = () => s.renderer.backend.device.queue.onSubmittedWorkDone();
				window.__poses = [];
				const afterRender = s.scene.onAfterRender;
				s.scene.onAfterRender = function ( ...args ) { window.__poses.push( s.camera.projectionMatrix.elements.slice() ); afterRender.apply( this, args ); };
				for ( let i = 0; i < 96; i ++ ) window.__step();
				await window.__drain(); window.__poses = [];
				for ( let i = 0; i < 16; i ++ ) window.__step();
				await window.__drain();
				return { distinctGeometryProjections: new Set( window.__poses.map( pose => JSON.stringify( pose ) ) ).size, traaOwnsJitter: s.renderPipeline._contextData.renderPipelineState.viewOffsetOwner === s.traaPass, camera: s.camera.position.toArray(), target: s.controls.target.toArray() };

			} );
			await page.screenshot( { path: `${ directory }/${ slug }-${ method }-stationary.png` } );
			const recovery = await page.evaluate( async () => {

				const s = window.__merge;
				const { DataUtils } = await import( '/src/Three.WebGPU.js' );
				const read = async () => {

					const target = s.traaPass._resolveRenderTarget;
					const data = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height );
					const result = Float32Array.from( data, DataUtils.fromHalfFloat );
					if ( ! result.every( Number.isFinite ) ) throw new Error( 'Non-finite AA output' );
					return result;

				};
				const distance = s.camera.position.distanceTo( s.controls.target );
				for ( let i = 0; i < 32; i ++ ) {

					s.camera.position.x += distance * 0.1 / 32;
					s.controls.target.x += distance * 0.1 / 32;
					window.__step();

				}
				await window.__drain();
				const samples = [];
				for ( let frame = 1; frame <= 64; frame ++ ) {

					window.__step();
					if ( [ 1, 2, 4, 8, 16, 32, 64 ].includes( frame ) ) samples.push( { frame, data: await read() } );

				}
				for ( let i = 0; i < 128; i ++ ) window.__step();
				const reference = new Float64Array( samples[ 0 ].data.length );
				for ( let frame = 0; frame < 32; frame ++ ) {

					window.__step(); const data = await read();
					for ( let i = 0; i < data.length; i ++ ) reference[ i ] += data[ i ] / 32;

				}
				return samples.map( sample => {

					let error = 0, count = 0;
					for ( let i = 0; i < reference.length; i ++ ) if ( i % 4 !== 3 ) { error += ( sample.data[ i ] - reference[ i ] ) ** 2; count ++; }
					return { frame: sample.frame, linearRGBRMSE: Math.sqrt( error / count ) };

				} );

			} );
			await page.screenshot( { path: `${ directory }/${ slug }-${ method }-settled-after-pan.png` } );
			for ( const view of [ 'SSGI Filtered', 'SSGI Filtered Coverage' ] ) {

				await page.evaluate( async view => { const s = window.__merge; s.params.output = view; s.updateOutput(); for ( let i = 0; i < 64; i ++ ) window.__step(); await window.__drain(); }, view );
				await page.screenshot( { path: `${ directory }/${ slug }-${ method }-${ view.toLowerCase().replaceAll( ' ', '-' ) }.png` } );

			}
			results.scenes.push( { method, model, diagnostic, recovery, errors } );
			console.log( method, model, diagnostic, recovery, errors );
			await writeFile( `${ directory }/results.json`, JSON.stringify( results, null, 2 ) );
			if ( errors.length ) throw new Error( 'Scene rendering errors' );
			page.off( 'pageerror', onError ); page.off( 'console', onConsole );

		}
		await page.close();

	}
	await writeFile( `${ directory }/results.json`, JSON.stringify( results, null, 2 ) );

} finally {

	await browser?.close(); await new Promise( resolve => server.close( resolve ) );

}

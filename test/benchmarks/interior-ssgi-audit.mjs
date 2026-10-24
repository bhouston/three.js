/* global window, document, navigator, performance */
// Run from the repository root with a hardware Chrome installation.
import puppeteer from 'puppeteer';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import { URL } from 'node:url';
import { createServer } from '../../utils/server.js';

const directory = process.env.SSGI_AUDIT_OUTPUT || '/tmp/three-ssgi-audit';
const baseline = process.env.SSGI_AUDIT_BASE || 'eedbf3e9833e';
const methods = ( process.env.SSGI_AUDIT_METHODS || 'baseline,current' ).split( ',' );
const models = process.env.SSGI_AUDIT_UNIT_ONLY === '1' ? [] : ( process.env.SSGI_AUDIT_MODELS || 'Cornell Box,Dual Rooms' ).split( ',' );
const files = [ 'examples/jsm/tsl/display/SSGINode.js', 'examples/jsm/tsl/display/CoverageDenoiseNode.js' ];
const original = Object.fromEntries( files.map( file => [ '/' + file, execFileSync( 'git-dedup', [ 'show', `${ baseline }:${ file }` ], { encoding: 'utf8' } ) ] ) );
const source = await readFile( 'examples/webgpu_interior.html', 'utf8' );
await mkdir( directory, { recursive: true } );
const server = createServer();
let browser;
const results = [];
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {

		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-background-timer-throttling', '--disable-renderer-backgrounding' ],
		defaultViewport: { width: 960, height: 720 }

	} );
	if ( process.env.SSGI_AUDIT_UNIT_ONLY === '1' ) {

		const page = await browser.newPage();
		const errors = [];
		page.on( 'pageerror', error => { errors.push( error.message ); console.error( error.message ); } );
		page.on( 'console', message => { if ( message.type() === 'error' ) { errors.push( message.text() ); console.error( message.text().slice( 0, 1600 ) ); } } );
		await page.setRequestInterception( true );
		page.on( 'request', request => request.url().endsWith( '/favicon.ico' ) ? request.respond( { status: 204 } ) : request.continue() );
		await page.goto( `http://localhost:${ server.address().port }/` );
		const hardware = await page.evaluate( async () => {

			const info = ( await navigator.gpu.requestAdapter() ).info;
			const gl = document.createElement( 'canvas' ).getContext( 'webgl2' );
			const debug = gl?.getExtension( 'WEBGL_debug_renderer_info' );
			const webglRenderer = debug ? gl.getParameter( debug.UNMASKED_RENDERER_WEBGL ) : null;
			return { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter, webglRenderer };

		} );
		if ( ! hardware.vendor || ! hardware.webglRenderer || hardware.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( hardware ) ) ) throw new Error( 'Hardware GPU required' );
		for ( const filter of ( process.env.SSGI_AUDIT_FILTERS || 'SSGINode,CoverageDenoiseNode' ).split( ',' ) ) {

			await page.goto( `http://localhost:${ server.address().port }/test/unit/UnitTestsAddons.html?filter=${ encodeURIComponent( filter ) }` );
			await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 120000 } );
			const result = await page.evaluate( () => ( {
				stats: window._QUnitStats,
				failures: Array.from( document.querySelectorAll( '#qunit-tests > .fail' ), element => element.textContent ),
				skipped: document.getElementById( 'qunit-tests' ).textContent.includes( 'SKIPPED' )
			} ) );
			results.push( { filter, ...result } );
			console.log( hardware, filter, result );

		}
		await writeFile( `${ directory }/unit-tests.json`, JSON.stringify( { hardware, results, errors }, null, 2 ) );
		if ( errors.length || results.some( result => result.stats.failed || ! result.stats.passed || result.skipped ) ) throw new Error( 'GPU tests failed, skipped, or produced rendering errors' );

	}
	for ( const model of models ) {

		for ( const method of methods ) {

			if ( ! [ 'baseline', 'current' ].includes( method ) ) throw new Error( 'Unsupported method' );
			const page = await browser.newPage();
			const errors = [];
			page.on( 'pageerror', error => { errors.push( error.message ); console.error( error.message ); } );
			page.on( 'console', message => { if ( message.type() === 'error' ) { errors.push( message.text() ); console.error( message.text().slice( 0, 1200 ) ); } } );
			const html = source.replace( "model: 'Dual Rooms'", `model: ${ JSON.stringify( model ) }` )
				.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
				.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
				.replace( 'renderer.setAnimationLoop( animate );', 'window.__audit = { renderer, camera, controls, params, ssgiParams, ssgiPass, ssgiDenoise, ssrPass, animate, updateOutput, updateSSGIParameters, resetScreenSpaceLighting }; renderer.setAnimationLoop( animate );' );
			await page.setRequestInterception( true );
			page.on( 'request', request => {

				const pathname = new URL( request.url() ).pathname;
				if ( pathname.endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
				if ( pathname === '/examples/webgpu_interior.html' ) return request.respond( { status: 200, contentType: 'text/html', body: html } );
				if ( method === 'baseline' && original[ pathname ] ) return request.respond( { status: 200, contentType: 'text/javascript', body: original[ pathname ] } );
				return request.continue();

			} );
			await page.goto( `http://localhost:${ server.address().port }/examples/webgpu_interior.html` );
			await page.waitForFunction( () => window.__audit && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 240000 } );
			const hardware = await page.evaluate( async () => {

				const adapter = await navigator.gpu.requestAdapter();
				const info = adapter.info;
				const hardware = { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
				if ( ! info.vendor || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( hardware ) ) ) throw new Error( 'Hardware GPU required' );
				const s = window.__audit;
				s.renderer.setAnimationLoop( null );
				s.renderer._animation.stop();
				s.controls.enableDamping = false;
				window.__auditStep = () => {

					s.renderer.info.reset();
					s.renderer._nodes.nodeFrame.update();
					s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId;
					s.animate();

				};
				window.__auditDrain = () => s.renderer.backend.device.queue.onSubmittedWorkDone();
				for ( let i = 0; i < 64; i ++ ) window.__auditStep();
				await window.__auditDrain();
				return hardware;

			} );
			console.log( model, method, hardware );
			await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
			const slug = model.toLowerCase().replaceAll( ' ', '-' );
			const captures = [];
			for ( const view of [ 'Beauty', 'SSGI Raw', 'SSGI Filtered', 'SSGI Filtered Coverage' ] ) {

				await page.evaluate( async view => {

					const s = window.__audit;
					s.params.output = view;
					s.updateOutput();
					for ( let i = 0; i < 64; i ++ ) window.__auditStep();
					await window.__auditDrain();

				}, view );
				const name = `${ slug }-${ method }-${ view.toLowerCase().replaceAll( ' ', '-' ) }.png`;
				await page.screenshot( { path: `${ directory }/${ name }` } );
				captures.push( name );

			}
			const throughput = await page.evaluate( async () => {

				const s = window.__audit;
				s.params.output = 'Beauty';
				s.updateOutput();
				for ( let i = 0; i < 64; i ++ ) window.__auditStep();
				await window.__auditDrain();
				const samples = [];
				for ( let repeat = 0; repeat < 3; repeat ++ ) {

					let completed = 0, drawCalls = 0;
					const initialGI = s.ssgiDenoise._frames;
					const initialSSR = s.ssrPass._noiseIndex.value;
					const start = performance.now();
					while ( performance.now() - start < 5000 ) {

						for ( let batch = 0; batch < 4; batch ++ ) {

							window.__auditStep();
							drawCalls += s.renderer.info.render.drawCalls;

						}
						await window.__auditDrain();
						completed += 4;

					}
					const seconds = ( performance.now() - start ) / 1000;
					if ( s.ssgiDenoise._frames - initialGI !== completed || s.ssrPass._noiseIndex.value - initialSSR !== completed ) throw new Error( 'Effect updates did not match completed frames' );
					samples.push( { completed, drawCalls, seconds, fps: completed / seconds } );

				}
				await window.__auditDrain();
				return samples;

			} );
			results.push( { model, method, hardware, captures, throughput, errors } );
			await writeFile( `${ directory }/results.json`, JSON.stringify( { baseline, viewport: [ 960, 720 ], lightMapResolution: 1024, results }, null, 2 ) );
			console.log( model, method, throughput );
			await page.close();
			if ( errors.length ) throw new Error( 'Rendering errors occurred' );

		}

	}

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

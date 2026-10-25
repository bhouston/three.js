/* global window, document, navigator */
// Hardware readback and hallway smoke checks for PRs 38–40. No throughput claims.
import puppeteer from 'puppeteer';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import { URL } from 'node:url';
import { createServer } from '../../utils/server.js';

const stage = process.env.INTEGRATION_STAGE || 'checkout';
const directory = process.env.INTEGRATION_OUTPUT || '/tmp/three-interior-pr-integration';
await mkdir( directory, { recursive: true } );
const server = createServer();
await new Promise( resolve => server.listen( 0, resolve ) );
const origin = `http://localhost:${ server.address().port }`;
let browser;
const result = { stage, units: [] };
try {

	browser = await puppeteer.launch( {
		executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-background-timer-throttling', '--disable-renderer-backgrounding' ],
		defaultViewport: { width: 800, height: 600 }
	} );
	const page = await browser.newPage();
	await page.setRequestInterception( true );
	page.on( 'request', request => {

		if ( new URL( request.url() ).pathname.endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
		return request.continue();

	} );
	const errors = [];
	page.on( 'pageerror', error => errors.push( error.message ) );
	page.on( 'console', message => { if ( message.type() === 'error' ) errors.push( message.text() ); } );
	await page.goto( origin );
	result.hardware = await page.evaluate( async () => {

		const adapter = await navigator.gpu.requestAdapter();
		if ( ! adapter ) throw new Error( 'No GPU adapter' );
		const info = adapter.info;
		const gl = document.createElement( 'canvas' ).getContext( 'webgl2' );
		const debug = gl.getExtension( 'WEBGL_debug_renderer_info' );
		return { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, fallback: info.isFallbackAdapter, webgl: gl.getParameter( debug.UNMASKED_RENDERER_WEBGL ) };

	} );
	if ( ! result.hardware.vendor || result.hardware.fallback || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( result.hardware ) ) ) throw new Error( 'Real GPU required' );
	console.log( 'Hardware', result.hardware );
	for ( const filter of [ 'SSGINode', 'CoverageDenoiseNode', 'TemporalPipeline', ...( stage === 'pr40' || stage === 'checkout' ? [ 'VelocityNode' ] : [] ) ] ) {

		const core = filter === 'VelocityNode';
		await page.goto( `${ origin }/test/unit/${ core ? 'UnitTests.html' : 'UnitTestsAddons.html' }?filter=${ filter }` );
		await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 120000 } );
		const unit = await page.evaluate( () => ( { stats: window._QUnitStats, failures: Array.from( document.querySelectorAll( '#qunit-tests > .fail' ), node => node.textContent ) } ) );
		result.units.push( { filter, ...unit } );
		console.log( filter, unit.stats );
		if ( unit.stats.failed || ! unit.stats.passed ) throw new Error( 'Regression suite failed' );

	}
	// A module document with the example import map, without loading/baking the example.
	const source = await readFile( 'examples/webgpu_interior.html', 'utf8' );
	page.removeAllListeners( 'request' );
	let html = '<!doctype html>' + source.match( /<script type="importmap">[\s\S]*?<\/script>/ )[ 0 ];
	page.on( 'request', request => {

		const path = new URL( request.url() ).pathname;
		if ( path === '/examples/webgpu_interior.html' ) return request.respond( { status: 200, contentType: 'text/html', body: html } );
		if ( path.endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
		return request.continue();

	} );
	await page.goto( `${ origin }/examples/webgpu_interior.html` );
	result.taau = await page.evaluate( async () => {

		const THREE = await import( 'three/webgpu' );
		const { pass, mrt, output, velocity, uv, vec3, float } = await import( 'three/tsl' );
		const { taau } = await import( 'three/addons/tsl/display/TAAUNode.js' );
		const renderer = new THREE.WebGPURenderer();
		await renderer.init(); renderer.setPixelRatio( 1 ); renderer.setSize( 64, 64 );
		const scene = new THREE.Scene();
		const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 ); camera.position.z = 2;
		const material = new THREE.MeshBasicNodeMaterial();
		material.colorNode = vec3( uv().x.mul( 24 ).fract().greaterThan( 0.8 ).select( float( 1 ), float( 0 ) ) );
		const geometry = new THREE.PlaneGeometry( 2, 2 ); scene.add( new THREE.Mesh( geometry, material ) );
		const beauty = pass( scene, camera ); beauty.setResolutionScale( 0.5 ); beauty.setMRT( mrt( { output, velocity } ) );
		const aa = taau( beauty, beauty.getTextureNode( 'depth' ), beauty.getTextureNode( 'velocity' ), camera );
		const pipeline = new THREE.RenderPipeline( renderer, aa );
		const step = async () => { renderer._nodes.nodeFrame.update(); pipeline.render(); await renderer.backend.device.queue.onSubmittedWorkDone(); };
		const read = async ( target, index ) => Array.from( await renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, index ), THREE.DataUtils.fromHalfFloat );
		const samples = [];
		for ( const size of [ 64, 80 ] ) {

			renderer.setSize( size, size );
			for ( let i = 0; i < 12; i ++ ) await step();
			const color = await read( aa._resolveRenderTarget, 0 );
			const history = await read( aa._historyRenderTarget, 1 );
			const lock = aa._resolveRenderTarget.textures.length > 1 ? await read( aa._resolveRenderTarget, 1 ) : null;
			samples.push( { size, finite: color.every( Number.isFinite ) && history.every( Number.isFinite ) && ( ! lock || lock.every( Number.isFinite ) ), nonzeroLocks: history.filter( ( v, i ) => i % 4 === 0 && v > 0 ).length, resolveMatchesHistory: lock ? lock.every( ( v, i ) => v === history[ i ] ) : false } );

		}
		material.colorNode = vec3( 0 ); material.needsUpdate = true;
		aa.setSize( 1, 1 );
		for ( let i = 0; i < 8; i ++ ) await step();
		const blackColor = await read( aa._resolveRenderTarget, 0 );
		const blackLock = await read( aa._historyRenderTarget, 1 );
		const black = { finite: blackColor.every( Number.isFinite ) && blackLock.every( Number.isFinite ), maxColor: Math.max( ...blackColor.filter( ( v, i ) => i % 4 !== 3 ) ), maxLock: Math.max( ...blackLock.filter( ( v, i ) => i % 4 === 0 ) ) };
		pipeline.dispose(); aa.dispose(); beauty.dispose();
		const targets = [ 0, 1 ].map( () => new THREE.RenderTarget( 64, 64, { count: 2, type: THREE.HalfFloatType } ) );
		for ( const target of targets ) { target.textures[ 0 ].name = 'output'; target.textures[ 1 ].name = 'velocity'; }
		renderer.setMRT( mrt( { output, velocity } ) );
		const motion = [];
		for ( let frame = 0; frame < 3; frame ++ ) {

			scene.children[ 0 ].position.x = frame * 0.1;
			renderer._nodes.nodeFrame.update(); renderer.info.frame = renderer._nodes.nodeFrame.frameId;
			const passes = [];
			for ( const target of targets ) {

				renderer.setRenderTarget( target ); renderer.render( scene, camera );
				await renderer.backend.device.queue.onSubmittedWorkDone();
				const data = await renderer.readRenderTargetPixelsAsync( target, 32, 32, 1, 1, 1 );
				passes.push( THREE.DataUtils.fromHalfFloat( data[ 0 ] ) );

			}
			motion.push( { frame, passes } );

		}
		renderer.setRenderTarget( null ); renderer.setMRT( null );
		for ( const target of targets ) target.dispose();
		geometry.dispose(); material.dispose(); renderer.dispose();
		return { samples, black, motion };

	} );
	console.log( 'TAAU', JSON.stringify( result.taau ) );
	if ( [ 'pr40', 'checkout' ].includes( stage ) && ! result.taau.motion.slice( 1 ).every( frame => frame.passes.every( value => Math.abs( value - 0.1 ) < 0.001 ) ) ) throw new Error( 'Multipass GPU velocity regression' );
	if ( ! result.taau.samples.every( s => s.finite ) || ! result.taau.black.finite ) throw new Error( 'Nonfinite TAAU output' );
	if ( stage !== 'baseline' && ! result.taau.samples.every( s => s.nonzeroLocks > 0 && s.resolveMatchesHistory ) ) throw new Error( 'Lock history regression' );
	if ( stage !== 'baseline' && stage !== 'pr38' && ( result.taau.black.maxColor !== 0 || result.taau.black.maxLock !== 0 ) ) throw new Error( 'Black neighborhood regression' );
	html = source.replace( "model: 'Dual Rooms'", "model: 'Space Ship Hallway'" )
		.replace( 'lightMapResolution: 4096', 'lightMapResolution: 1024' )
		.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', 'renderer.setPixelRatio( 1 );' )
		.replace( 'renderer.setAnimationLoop( animate );', 'window.__integration = { renderer, camera, controls, params, scene, prePass, traaPass, renderPipeline, animate }; renderer.setAnimationLoop( animate );' );
	await page.goto( `${ origin }/examples/webgpu_interior.html` );
	console.log( 'Baking hallway' );
	await page.waitForFunction( () => window.__integration && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 900000 } );
	result.hallway = await page.evaluate( async () => {

		const s = window.__integration;
		s.renderer.setAnimationLoop( null ); s.renderer._animation.stop(); s.controls.enableDamping = false;
		const step = () => { s.renderer.info.reset(); s.renderer._nodes.nodeFrame.update(); s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId; s.animate(); };
		const drain = () => s.renderer.backend.device.queue.onSubmittedWorkDone();
		for ( let i = 0; i < 48; i ++ ) { step(); if ( i % 8 === 7 ) await drain(); }
		await drain();
		const { DataUtils } = await import( 'three/webgpu' );
		const read = async () => {

			const target = s.traaPass._resolveRenderTarget;
			const data = Array.from( await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height ), DataUtils.fromHalfFloat );
			return { finite: data.every( Number.isFinite ), nonzero: data.filter( ( v, i ) => i % 4 !== 3 && v > 0 ).length };

		};
		const stationary = await read();
		for ( let i = 0; i < 16; i ++ ) { s.camera.position.x += 0.002; s.controls.target.x += 0.002; step(); }
		await drain();
		for ( let i = 0; i < 32; i ++ ) step();
		await drain();
		return { model: s.params.model, status: document.getElementById( 'status' ).textContent, stationary, afterMotion: await read(), traaOwnsJitter: s.renderPipeline._contextData.renderPipelineState.viewOffsetOwner === s.traaPass };

	} );
	await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
	await page.screenshot( { path: `${ directory }/${ stage }-hallway.png` } );
	result.errors = errors;
	console.log( 'Hallway', result.hallway, 'Errors', errors );
	if ( ! result.hallway.stationary.finite || ! result.hallway.afterMotion.finite || ! result.hallway.stationary.nonzero || ! result.hallway.traaOwnsJitter || errors.length ) throw new Error( 'Hallway regression' );

} finally {

	await writeFile( `${ directory }/${ stage }.json`, JSON.stringify( result, null, 2 ) );
	await browser?.close(); await new Promise( resolve => server.close( resolve ) );

}

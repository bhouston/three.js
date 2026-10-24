/* global window, navigator, requestAnimationFrame */
// Compare the full temporal dependency chain to an independent supersampled raster reference.
// Run from the repository root; no timing/throughput claims are made.
import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';
import { execFileSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import { URL } from 'node:url';
import { createServer } from '../../utils/server.js';

const directory = process.env.TEMPORAL_AA_OUTPUT || '/tmp/three-merge-aa-reference';
const baseline = process.env.TEMPORAL_AA_BASE || '876361339e40';
const width = 256, height = 192;
const files = [ 'examples/jsm/tsl/display/TRAANode.js', 'examples/jsm/tsl/display/TemporalReprojectNode.js', 'examples/jsm/tsl/utils/TAAUtils.js' ];
const originals = Object.fromEntries( files.map( file => [ '/' + file, execFileSync( 'git-dedup', [ 'show', `${ baseline }:${ file }` ], { encoding: 'utf8' } ) ] ) );
const html = `<!doctype html><script type="importmap">{"imports":{"three":"/src/Three.js","three/webgpu":"/src/Three.WebGPU.js","three/tsl":"/src/Three.TSL.js"}}</script><script type="module">window.ready=true;</script>`;
await mkdir( directory, { recursive: true } );
const server = createServer();
let browser;
const results = [];
const errors = [];
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit' ], defaultViewport: { width, height } } );
	for ( const fixture of [ 'stress', 'dependency' ] ) for ( const method of [ 'baseline', 'current' ] ) {

		const page = await browser.newPage();
		page.on( 'pageerror', error => errors.push( { method, message: error.message } ) );
		page.on( 'console', message => { if ( message.type() === 'error' ) errors.push( { method, message: message.text() } ); } );
		await page.setRequestInterception( true );
		page.on( 'request', request => {

			const pathname = new URL( request.url() ).pathname;
			if ( pathname === '/aa-reference' ) return request.respond( { status: 200, contentType: 'text/html', body: html } );
			if ( pathname.endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
			if ( method === 'baseline' && originals[ pathname ] ) return request.respond( { status: 200, contentType: 'text/javascript', body: originals[ pathname ] } );
			return request.continue();

		} );
		await page.goto( `http://localhost:${ server.address().port }/aa-reference` );
		await page.waitForFunction( () => window.ready );
		const captures = await page.evaluate( async ( { width, height, fixture } ) => {

			const THREE = await import( 'three/webgpu' );
			const { convertToTexture, mrt, output, pass, uniform, vec4, velocity } = await import( 'three/tsl' );
			const { traa } = await import( '/examples/jsm/tsl/display/TRAANode.js' );
			const { temporalReproject } = await import( '/examples/jsm/tsl/display/TemporalReprojectNode.js' );
			const renderer = new THREE.WebGPURenderer( { antialias: false } );
			await renderer.init();
			const info = ( await navigator.gpu.requestAdapter() ).info;
			const hardware = { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter, backend: renderer.backend.constructor.name };
			if ( ! renderer.backend.isWebGPUBackend || ! info.vendor || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( hardware ) ) ) throw new Error( 'Real hardware WebGPU required' );
			renderer.setPixelRatio( 1 );
			renderer.setSize( width, height );
			renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
			renderer.toneMapping = THREE.NoToneMapping;
			const camera = new THREE.OrthographicCamera( - 2, 2, 1.5, - 1.5, 0.1, 20 );
			camera.position.z = 5;
			camera.coordinateSystem = renderer.coordinateSystem;
			camera.updateProjectionMatrix();
			camera.updateMatrixWorld();
			const scene = new THREE.Scene();
			scene.background = new THREE.Color( 0 );
			const material = new THREE.MeshBasicNodeMaterial( { color: 0xffffff } );
			const addPlane = ( x, y, w, h, angle ) => {

				const mesh = new THREE.Mesh( new THREE.PlaneGeometry( w, h ), material );
				mesh.position.set( x, y, 0 );
				mesh.rotation.z = angle;
				scene.add( mesh );

			};
			addPlane( - 0.85, 0.1, 1.15, 1.75, 0.371 );
			addPlane( 0.75, 0.7, 1.3, 0.018, - 0.233 );
			addPlane( 0.8, 0.2, 1.4, 0.037, 0.619 );
			addPlane( 0.7, - 0.5, 1.5, 0.09, - 0.397 );
			const target = new THREE.RenderTarget( width, height, { type: THREE.FloatType, depthBuffer: false } );
			const source = pass( scene, camera );
			source.setMRT( mrt( { output, normal: vec4( 0.5, 0.5, 1, 1 ), velocity } ) );
			const temporal = temporalReproject( source, source.getTextureNode( 'depth' ), source.getTextureNode( 'normal' ), source.getTextureNode( 'velocity' ), camera, { accumulate: true } );
			// A runtime zero uniform retains the upstream dependency without filtering beauty.
			const aaInput = fixture === 'stress' ? temporal : convertToTexture( source.getTextureNode().add( temporal.mul( uniform( 0 ) ) ) );
			const aa = traa( aaInput, source.getTextureNode( 'depth' ), source.getTextureNode( 'velocity' ), camera );
			aa.useSubpixelCorrection = false;
			const pipeline = new THREE.RenderPipeline( renderer, aa );
			const projections = [];
			scene.onAfterRender = () => projections.push( camera.projectionMatrix.elements.slice() );
			const frames = [];
			for ( let frame = 1; frame <= 96; frame ++ ) {

				await new Promise( resolve => requestAnimationFrame( resolve ) );
				renderer.setRenderTarget( target );
				pipeline.render();
				if ( frame === 64 || frame === 96 ) frames.push( { frame, pixels: Array.from( await renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height ) ) } );

			}
			const velocityIndex = source.renderTarget.textures.indexOf( source.getTexture( 'velocity' ) );
			const motion = await renderer.readRenderTargetPixelsAsync( source.renderTarget, 0, 0, width, height, velocityIndex );
			let maximumMotion = 0;
			for ( let i = 0; i < motion.length; i += 4 ) maximumMotion = Math.max( maximumMotion, Math.abs( THREE.DataUtils.fromHalfFloat( motion[ i ] ) ), Math.abs( THREE.DataUtils.fromHalfFloat( motion[ i + 1 ] ) ) );
			await renderer.backend.device.queue.onSubmittedWorkDone();
			const distinctProjectionCount = new Set( projections.slice( - 64 ).map( value => JSON.stringify( value ) ) ).size;
			const jitterOwner = pipeline._contextData.renderPipelineState.viewOffsetOwner === aa ? 'TRAA' : 'upstream';
			// Independent ordinary rasterization, without either temporal pass.
			scene.onAfterRender = () => {};
			const referenceTarget = new THREE.RenderTarget( width * 4, height * 4, { type: THREE.FloatType } );
			renderer.setSize( width * 4, height * 4 );
			renderer.setRenderTarget( referenceTarget );
			renderer.render( scene, camera );
			const high = await renderer.readRenderTargetPixelsAsync( referenceTarget, 0, 0, width * 4, height * 4 );
			const reference = new Array( width * height * 4 ).fill( 0 );
			for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) for ( let sy = 0; sy < 4; sy ++ ) for ( let sx = 0; sx < 4; sx ++ ) for ( let c = 0; c < 4; c ++ ) reference[ 4 * ( x + width * y ) + c ] += high[ 4 * ( x * 4 + sx + width * 4 * ( y * 4 + sy ) ) + c ] / 16;
			await renderer.backend.device.queue.onSubmittedWorkDone();
			renderer.dispose();
			return { hardware, frames, reference, distinctProjectionCount, jitterOwner, maximumMotion };

		}, { width, height, fixture } );
		const { reference, frames, ...metadata } = captures;
		const metrics = pixels => {

			let squaredError = 0, edgeError = 0, edgeCount = 0, gradientEnergy = 0, referenceGradientEnergy = 0;
			for ( let y = 0; y < height; y ++ ) for ( let x = 0; x < width; x ++ ) {

				const i = 4 * ( x + width * y );
				const error = pixels[ i ] - reference[ i ];
				squaredError += error * error;
				let edge = false;
				for ( let oy = - 1; oy <= 1; oy ++ ) for ( let ox = - 1; ox <= 1; ox ++ ) {

					const j = 4 * ( Math.max( 0, Math.min( width - 1, x + ox ) ) + width * Math.max( 0, Math.min( height - 1, y + oy ) ) );
					if ( reference[ j ] > 0 && reference[ j ] < 1 ) edge = true;

				}
				if ( edge ) { edgeError += error * error; edgeCount ++; }
				for ( const j of [ x + 1 < width ? i + 4 : i, y + 1 < height ? i + 4 * width : i ] ) {

					gradientEnergy += ( pixels[ j ] - pixels[ i ] ) ** 2;
					referenceGradientEnergy += ( reference[ j ] - reference[ i ] ) ** 2;

				}

			}
			return { linearRGBRMSE: Math.sqrt( squaredError / ( width * height ) ), referenceEdgeBandRMSE: Math.sqrt( edgeError / edgeCount ), gradientEnergyRatio: gradientEnergy / referenceGradientEnergy, edgePixelCount: edgeCount };

		};
		const saveImage = async ( name, pixels ) => {

			const png = new PNG( { width, height } );
			for ( let i = 0; i < pixels.length; i ++ ) {

				const v = Math.max( 0, Math.min( 1, pixels[ i ] ) );
				png.data[ i ] = i % 4 === 3 ? 255 : Math.round( 255 * ( v <= 0.0031308 ? 12.92 * v : 1.055 * v ** ( 1 / 2.4 ) - 0.055 ) );

			}
			await writeFile( `${ directory }/${ name }.png`, PNG.sync.write( png ) );

		};
		await saveImage( `${ fixture }-${ method }-reference-4x`, reference );
		const frameResults = [];
		for ( const frame of frames ) {

			await saveImage( `${ fixture }-${ method }-${ frame.frame }`, frame.pixels );
			frameResults.push( { frame: frame.frame, ...metrics( frame.pixels ) } );

		}
		results.push( { fixture, method, ...metadata, frames: frameResults } );
		console.log( JSON.stringify( results.at( - 1 ) ) );
		await page.close();

	}
	await writeFile( `${ directory }/results.json`, JSON.stringify( { baseline, width, height, reference: '4x per axis ordinary rasterization, box downsampled in linear RGB; synthetic unlit geometry. Stress: entire Beauty passes through temporal reprojection then TRAA. Dependency: Beauty plus runtime-zero temporal dependency before TRAA, preserving original sharp input' , limitations: 'Check recorded ownership per fixture; a simple temporal input need not reproduce the baseline jitter-owner defect. Stress double-filters whole Beauty whereas the interior denoises indirect lighting only, so it is not a prediction of interior quality. Dependency fixture preserves sharp Beauty input. Neither establishes physical lighting correctness.', gradientMetric: 'Squared horizontal/vertical luminance-gradient energy divided by reference energy; below1 may indicate extra blur, above1 excess high-frequency energy; not a standalone quality score', results, errors }, null, 2 ) );
	if ( errors.length ) throw new Error( JSON.stringify( errors ) );

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

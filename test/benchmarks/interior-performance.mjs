/* global window, document, navigator, performance, Event */
// Hardware-only sustained full-pipeline throughput and separate Inspector diagnostics.
import puppeteer from 'puppeteer';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import { createServer } from '../../utils/server.js';

const directory = process.env.INTERIOR_PROFILE_OUTPUT || '/tmp/three-interior-profile-20261024';
const models = ( process.env.INTERIOR_PROFILE_MODELS || 'Space Ship Hallway,Dungeon Warkarma' ).split( ',' );
const variants = ( process.env.INTERIOR_PROFILE_VARIANTS || 'baseline,ssr-half,ssr-quarter,no-ssr,no-ssgi,no-ao,no-screen-effects,half-pixels,baseline-repeat' ).split( ',' );
const unitFilters = ( process.env.INTERIOR_PROFILE_UNIT_FILTERS || '' ).split( ',' ).filter( Boolean );
const validate = process.env.INTERIOR_PROFILE_VALIDATE === '1';
const pixelRatio = Number( process.env.INTERIOR_PROFILE_PIXEL_RATIO || 1 );
const seconds = Number( process.env.INTERIOR_PROFILE_SECONDS || 5 );
const repeats = Number( process.env.INTERIOR_PROFILE_REPEATS || 3 );
const supported = [ 'baseline', 'baseline-repeat', 'ssr-full', 'ssr-half', 'ssr-quarter', 'ssr-filter-4', 'ssr-quality-half', 'ssr-half-quality-half', 'ssr-no-denoise', 'ssgi-no-denoise', 'no-ssr', 'no-ssgi', 'no-ao', 'no-screen-effects', 'half-pixels', 'prepass-unlit', 'merge-static' ];
if ( variants.some( variant => ! supported.includes( variant ) ) || ! Number.isFinite( pixelRatio ) || pixelRatio <= 0 || ! Number.isFinite( seconds ) || seconds < 5 || ! Number.isInteger( repeats ) || repeats < 1 ) throw new Error( 'Use supported variants, at least five seconds and a positive repeat count' );
const source = ( await readFile( 'examples/webgpu_interior.html', 'utf8' ) ).replace( 'cache: diffuse', process.env.INTERIOR_PROFILE_SOURCE_CACHE === '0' ? 'cache: false' : 'cache: diffuse' );
await mkdir( directory, { recursive: true } );
const results = { viewport: { width: 1280, height: 720, deviceScaleFactor: pixelRatio }, seconds, repeats, pixelRatio, bakeAtlas: 4096, batchFrames: 4, warmupMs: 2500, inspectorDuringThroughput: false, scenes: [] };
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit', '--disable-background-timer-throttling', '--disable-renderer-backgrounding' ],
		defaultViewport: results.viewport
	} );
	results.browser = await browser.version();
	if ( unitFilters.length ) {

		const page = await browser.newPage();
		await page.goto( `http://localhost:${ server.address().port }/` );
		results.hardware = await page.evaluate( async () => {

			const info = ( await navigator.gpu.requestAdapter() ).info;
			if ( ! info.vendor || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( info ) ) ) throw new Error( 'Hardware GPU required' );
			return { vendor: info.vendor, architecture: info.architecture, isFallbackAdapter: info.isFallbackAdapter };

		} );
		results.units = [];
		for ( const filter of unitFilters ) {

			await page.goto( `http://localhost:${ server.address().port }/test/unit/UnitTestsAddons.html?filter=${ encodeURIComponent( filter ) }` );
			await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 180000 } );
			const result = await page.evaluate( () => ( { stats: window._QUnitStats, failures: Array.from( document.querySelectorAll( '#qunit-tests > .fail' ), element => element.textContent ), skipped: document.getElementById( 'qunit-tests' ).textContent.includes( 'SKIPPED' ) } ) );
			results.units.push( { filter, ...result } );
			console.log( filter, result );
			await writeFile( `${ directory }/results.json`, JSON.stringify( results, null, 2 ) );
			if ( result.stats.failed || ! result.stats.passed || result.skipped ) throw new Error( 'GPU unit tests failed or skipped' );

		}
		await page.close();

	}
	for ( const model of unitFilters.length ? [] : models ) {

		const page = await browser.newPage();
		const errors = [];
		page.on( 'pageerror', error => { errors.push( error.message ); console.error( error.message ); } );
		page.on( 'console', message => { if ( message.type() === 'error' ) { errors.push( message.text() ); console.error( message.text().slice( 0, 1000 ) ); } } );
		const html = source.replace( "model: 'Dual Rooms'", `model: ${ JSON.stringify( model ) }` )
			.replace( 'renderer.setPixelRatio( window.devicePixelRatio );', `renderer.setPixelRatio( ${ pixelRatio } );` )
			.replace( 'renderer.setAnimationLoop( animate );', 'window.__profile = { renderer, camera, scene, model, controls, params, ssrParams, ssgiParams, prePass, scenePass, traaPass, renderPipeline, animate, updateRenderSize, resetScreenSpaceLighting, updateAO, updateOutput, updateSSRParameters, updateSSGIParameters, updateIndirectLighting }; renderer.setAnimationLoop( animate );' );
		await page.setRequestInterception( true );
		page.on( 'request', request => request.url().endsWith( '/favicon.ico' ) ? request.respond( { status: 204 } ) : request.url().split( '?' )[ 0 ].endsWith( '/examples/webgpu_interior.html' ) ? request.respond( { status: 200, contentType: 'text/html', body: html } ) : request.continue() );
		console.log( 'Loading and baking', model );
		await page.goto( `http://localhost:${ server.address().port }/examples/webgpu_interior.html` );
		await page.waitForFunction( () => window.__profile && document.getElementById( 'status' ).textContent.includes( 'complete' ), { timeout: 900000 } );
		const metadata = await page.evaluate( async () => {

			const s = window.__profile;
			s.renderer.setAnimationLoop( null ); s.renderer._animation.stop();
			await s.renderer.inspector.resolveTimestamp();
			const adapter = await navigator.gpu.requestAdapter();
			const info = adapter.info;
			const gl = document.createElement( 'canvas' ).getContext( 'webgl2' );
			const debug = gl.getExtension( 'WEBGL_debug_renderer_info' );
			const hardware = { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter, webglRenderer: gl.getParameter( debug.UNMASKED_RENDERER_WEBGL ), timestampQuery: s.renderer.hasFeature( 'timestamp-query' ), device: s.renderer.backend.device.adapterInfo, deviceLabel: s.renderer.backend.device.label };
			if ( ! info.vendor || info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( hardware ) ) ) throw new Error( 'Hardware GPU required' );
			gl.getExtension( 'WEBGL_lose_context' ).loseContext();
			s.controls.enableDamping = false;
			if ( s.params.model === 'Dungeon Warkarma' ) {

				// Match webgpu_postprocessing_ssr_denoise, including its 0.1 model scale.
				s.camera.position.set( 1.259878548682251, 0.5391287340899181, - 0.27217301481427114 );
				s.controls.target.set( 1.0258536154689288, 0.2746440590977971, - 1.0815876858987743 );
				s.camera.fov = 35; s.camera.near = 0.1; s.camera.far = 8;
				s.camera.updateProjectionMatrix(); s.controls.update();

			}
			window.__defaults = { pixelRatio: s.renderer.getPixelRatio(), params: { ...s.params }, ssr: { ...s.ssrParams }, ssgi: { ...s.ssgiParams } };
			window.__step = ( profile = false ) => { s.renderer.info.reset(); s.renderer._nodes.nodeFrame.update(); s.renderer.info.frame = s.renderer._nodes.nodeFrame.frameId; if ( profile ) s.renderer.inspector.begin(); s.animate(); if ( profile ) s.renderer.inspector.finish(); };
			window.__drain = () => s.renderer.backend.device.queue.onSubmittedWorkDone();
			let meshes = 0, triangles = 0, lights = 0;
			const materials = new Set();
			s.scene.traverse( object => { if ( object.isMesh ) { meshes ++; triangles += ( object.geometry.index?.count || object.geometry.attributes.position.count ) / 3; for ( const material of Array.isArray( object.material ) ? object.material : [ object.material ] ) materials.add( material ); } if ( object.isLight ) lights ++; } );
			return { hardware, meshes, triangles, materials: materials.size, lights, camera: s.camera.position.toArray(), target: s.controls.target.toArray(), fov: s.camera.fov, near: s.camera.near, far: s.camera.far, loadedModel: s.params.model, bakeStatus: document.getElementById( 'status' ).textContent };

		} );
		if ( metadata.loadedModel !== model ) throw new Error( 'Requested model was not loaded' );
		console.log( 'Baked', model, metadata );
		await page.addStyleTag( { content: 'body > :not(canvas):not(script):not(style) { display: none !important; }' } );
		const entry = { model, ...metadata, variants: [], errors };
		results.scenes.push( entry );
		for ( const variant of variants ) {

			await page.evaluate( async variant => {

				const s = window.__profile;
				await window.__drain();
				Object.assign( s.params, window.__defaults.params ); Object.assign( s.ssrParams, window.__defaults.ssr ); Object.assign( s.ssgiParams, window.__defaults.ssgi );
				if ( s.merged ) s.merged.visible = false;
				s.model.visible = true;
				if ( variant === 'merge-static' ) {

					if ( ! s.merged ) {

						const { Group, Mesh } = await import( '/src/Three.WebGPU.js' );
						const { mergeGeometries } = await import( '/examples/jsm/utils/BufferGeometryUtils.js' );
						const groups = new Map();
						s.model.updateMatrixWorld( true );
						s.model.traverseVisible( object => {

							if ( ! object.isMesh ) return;
							if ( object.isSkinnedMesh || object.isInstancedMesh || Array.isArray( object.material ) || object.material.transparent || object.geometry.groups.length || Object.keys( object.geometry.morphAttributes ).length || object.matrixWorld.determinant() < 0 ) throw new Error( 'Unsupported mesh for static merge experiment' );
							const key = object.material.uuid + ':' + Object.keys( object.geometry.attributes ).sort().join( ',' ) + ':' + Boolean( object.geometry.index );
							if ( ! groups.has( key ) ) groups.set( key, { material: object.material, geometries: [] } );
							groups.get( key ).geometries.push( object.geometry.clone().applyMatrix4( object.matrixWorld ) );

						} );
						s.merged = new Group();
						for ( const { material, geometries } of groups.values() ) {

							const geometry = mergeGeometries( geometries );
							if ( ! geometry ) throw new Error( 'Static merge failed' );
							const mesh = new Mesh( geometry, material ); mesh.castShadow = true; mesh.receiveShadow = true;
							s.merged.add( mesh );
							for ( const source of geometries ) source.dispose();

						}
						s.scene.add( s.merged );

					}
					s.model.visible = false; s.merged.visible = true;

				}
				s.prePass.lighting = null;
				if ( variant === 'prepass-unlit' ) {

					const { Lighting } = await import( '/src/Three.WebGPU.js' );
					s.prePass.lighting = new Lighting(); s.prePass.lighting.enabled = false;

				}
				s.prePass.needsUpdate = true;
				s.renderer.setPixelRatio( window.__defaults.pixelRatio * ( variant === 'half-pixels' ? Math.SQRT1_2 : 1 ) );
				if ( variant === 'ssr-filter-4' ) s.ssrParams.denoiseSamples = 4;
				if ( variant === 'ssr-full' ) s.ssrParams.resolutionScale = 1;
				if ( variant === 'ssr-half' ) s.ssrParams.resolutionScale = 0.5;
				if ( variant === 'ssr-quarter' ) s.ssrParams.resolutionScale = 0.25;
				if ( variant === 'ssr-quality-half' ) s.ssrParams.quality = 0.5;
				if ( variant === 'ssr-half-quality-half' ) { s.ssrParams.resolutionScale = 0.5; s.ssrParams.quality = 0.5; }
				if ( variant === 'ssr-no-denoise' ) s.ssrParams.denoise = false;
				if ( variant === 'ssgi-no-denoise' ) s.ssgiParams.denoise = false;
				if ( variant === 'no-ssr' || variant === 'no-screen-effects' ) s.ssrParams.enabled = false;
				if ( variant === 'no-ssgi' || variant === 'no-screen-effects' ) s.ssgiParams.enabled = false;
				if ( variant === 'no-ao' || variant === 'no-screen-effects' ) s.params.aoType = 'None';
				s.updateSSRParameters(); s.updateSSGIParameters(); s.updateIndirectLighting();
				// Keep Inspector installed, but turn off its collection and GPU queries during throughput.
				s.renderer.inspector.enabled = false; s.renderer.inspector.lastFrame = null; s.renderer.backend.trackTimestamp = false;
				const warmupStart = performance.now();
				do {

					for ( let i = 0; i < 4; i ++ ) window.__step();
					await window.__drain();

				} while ( performance.now() - warmupStart < 2500 );

			}, variant );
			const measurements = await page.evaluate( async ( { seconds, repeats } ) => {

				const s = window.__profile;
				const samples = [];
				for ( let repeat = 0; repeat < repeats; repeat ++ ) {

					let completed = 0, drawCalls = 0;
					const start = performance.now();
					while ( performance.now() - start < seconds * 1000 ) {

						for ( let i = 0; i < 4; i ++ ) { window.__step(); drawCalls += s.renderer.info.render.drawCalls; }
						await window.__drain(); completed += 4;

					}
					const elapsedMs = performance.now() - start;
					samples.push( { completed, elapsedMs, fps: completed * 1000 / elapsedMs, drawCallsPerFrame: drawCalls / completed } );

				}
				return samples;

			}, { seconds, repeats } );
			// GPU/CPU timestamps are a separate diagnostic, never used as throughput measurements.
			const passes = await page.evaluate( async () => {

				const s = window.__profile;
				const inspector = s.renderer.inspector;
				inspector.enabled = true; s.renderer.backend.trackTimestamp = true; inspector.statsData.clear();
				for ( let i = 0; i < 24; i ++ ) { window.__step( true ); await window.__drain(); await inspector.resolveTimestamp(); }
				const recent = inspector.frames.slice( - 24 );
				const groups = new Map();
				for ( const frame of recent ) for ( const pass of [ ...frame.renders, ...frame.computes ] ) {

					const key = pass.cid;
					if ( ! groups.has( key ) ) groups.set( key, { id: key, name: pass.name, cpu: [], gpu: [], parent: pass.parent?.cid || null } );
					const group = groups.get( key );
					group.cpu.push( pass.cpu );
					if ( frame.resolvedRender && ! pass.gpuNotAvailable ) group.gpu.push( pass.gpu );

				}
				const average = values => values.length ? values.reduce( ( a, b ) => a + b, 0 ) / values.length : null;
				const data = Array.from( groups.values(), group => ( { id: group.id, name: group.name, parent: group.parent, cpuMs: average( group.cpu ), gpuMs: average( group.gpu ), gpuSamples: group.gpu.length } ) );
				inspector.enabled = false; inspector.lastFrame = null; s.renderer.backend.trackTimestamp = false;
				return data;

			} );
			await page.screenshot( { path: `${ directory }/${ model.replaceAll( ' ', '-' ) }-${ variant }.png` } );
			entry.variants.push( { variant, measurements, passes } );
			await writeFile( `${ directory }/results.json`, JSON.stringify( results, null, 2 ) );
			console.log( model, variant, measurements.map( sample => sample.fps.toFixed( 2 ) ), 'fps' );
			if ( errors.length ) throw new Error( 'Scene reported browser or GPU errors; inspect results.json' );

		}
		if ( validate ) {

			entry.validation = await page.evaluate( async reducedFilter => {

				const s = window.__profile;
				const { DataUtils } = await import( '/src/Three.WebGPU.js' );
				const snapshots = [];
				for ( const scale of [ 1, 0.5, 0.8, 1 ] ) {

					s.params.renderScale = scale; s.updateRenderSize();
					window.dispatchEvent( new Event( 'resize' ) );
					for ( let i = 0; i < 16; i ++ ) { window.__step(); await window.__drain(); }
					const width = Math.floor( window.innerWidth * window.devicePixelRatio * scale );
					const height = Math.floor( window.innerHeight * window.devicePixelRatio * scale );
					if ( s.renderer.domElement.width !== width || s.renderer.domElement.height !== height ) throw new Error( 'Incorrect render scale' );
					if ( s.camera.aspect !== window.innerWidth / window.innerHeight ) throw new Error( 'Render scale changed camera aspect' );
					const target = s.traaPass._resolveRenderTarget;
					if ( target.width !== width || target.height !== height ) throw new Error( 'Temporal target did not resize' );
					const pixels = await s.renderer.readRenderTargetPixelsAsync( target, 0, 0, width, height );
					if ( ! pixels.every( value => Number.isFinite( DataUtils.fromHalfFloat( value ) ) ) ) throw new Error( 'Non-finite reconstructed output' );
					snapshots.push( { scale, width, height } );

				}
				// Exercise camera motion and recovery with both native and reduced SSR tracing.
				const position = s.camera.position.clone(), target = s.controls.target.clone();
				if ( reducedFilter ) s.ssrParams.denoiseSamples = 4;
				for ( const resolutionScale of [ 1, 0.5 ] ) {

					s.ssrParams.resolutionScale = resolutionScale; s.updateSSRParameters();
					for ( let i = 0; i < 32; i ++ ) { s.camera.position.x += 0.0005; s.controls.target.x += 0.0005; window.__step(); await window.__drain(); }
					for ( let i = 0; i < 32; i ++ ) { window.__step(); await window.__drain(); }

				}
				s.camera.position.copy( position ); s.controls.target.copy( target ); s.ssrParams.resolutionScale = window.__defaults.ssr.resolutionScale; s.updateSSRParameters();
				return { snapshots, cameraMotionFrames: 64, recoveryFrames: 64, denoiseSamples: s.ssrParams.denoiseSamples };

			}, variants.includes( 'ssr-filter-4' ) );
			await page.screenshot( { path: `${ directory }/${ model.replaceAll( ' ', '-' ) }-after-motion.png` } );
			await writeFile( `${ directory }/results.json`, JSON.stringify( results, null, 2 ) );
			console.log( 'Validated scale changes, resize and camera motion', model );
			if ( errors.length ) throw new Error( 'Validation reported browser or GPU errors' );

		}
		await page.close();

	}

} finally {

	await browser?.close(); await new Promise( resolve => server.close( resolve ) );

}

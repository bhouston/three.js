// Hardware-only SSR quality comparison. Run after `npm run build`:
// node test/benchmarks/ssr-quality.js --baseline=<revision> --chrome=<executable> --output=<directory>
// The baseline revision supplies CoverageDenoiseNode only; scene, sampling,
// resolution, exposure, source and AA are identical in the filter comparison.
import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createServer } from '../../utils/server.js';

const option = name => process.argv.find( value => value.startsWith( `--${ name }=` ) )?.split( '=' ).slice( 1 ).join( '=' );
const baseline = option( 'baseline' );
const output = path.resolve( option( 'output' ) || '/tmp/three-ssr-quality' );
const chrome = option( 'chrome' );
const width = 640, height = 480;
const scenarios = [ 'static', 'camera-translation', 'grazing-floor', 'thin-objects', 'bright-emitter', 'moving-reflected-object', 'screen-edge-exit' ];
const root = path.resolve( import.meta.dirname, '../..' );
const html = await readFile( path.join( root, 'examples/webgpu_postprocessing_ssr.html' ), 'utf8' );
const originalDenoiser = baseline ? execFileSync( 'git-dedup', [ 'show', `${ baseline }:examples/jsm/tsl/display/CoverageDenoiseNode.js` ], { cwd: root, encoding: 'utf8' } ) : null;
const source = html.replace( 'renderer.setAnimationLoop( animate );', `
			// Test-only control: the shipped example has no automation globals.
			renderer._animation.stop();
			controls.enableDamping = false;
			const fixtures = new THREE.Group();
			scene.add( fixtures );
			const rods = new THREE.Group();
			for ( let i = 0; i < 5; i ++ ) {

				const rod = new THREE.Mesh( new THREE.CylinderGeometry( 0.015, 0.015, 1.4, 12 ), new THREE.MeshStandardMaterial( { color: 0xeeeeee, roughness: 0.15, metalness: 0.3 } ) );
				rod.position.set( - 0.8 + i * 0.2, - 0.05, 0.4 );
				rods.add( rod );

			}
			const emitter = new THREE.Mesh( new THREE.BoxGeometry( 0.2, 0.5, 0.2 ), new THREE.MeshBasicMaterial( { color: new THREE.Color( 16, 8, 2 ) } ) );
			emitter.position.set( 0.8, - 0.2, 0.2 );
			const moving = new THREE.Mesh( new THREE.BoxGeometry( 0.35, 0.5, 0.35 ), new THREE.MeshStandardMaterial( { color: 0xff2211, roughness: 0.15, metalness: 0.5 } ) );
			fixtures.add( rods, emitter, moving );
			let activeScenario = 'static';
			window.__ssrQuality = {
				ready: () => Boolean( model ),
				renderer, camera, scene, controls, params, ssrPass, ssrDenoise,
				configure( options ) {

					Object.assign( params, options );
					params.mirrorBias = 0;
					renderer.toneMappingExposure = 1;
					useHistorySource.value = params.reflectionSource === 'Previous frame';
					ssrPass.setColorNode( useHistorySource.value ? previousTraceSource : currentTraceSource );
					views.Beauty = params.antialiasing === 'TRAA' ? traaPass : smaaPass;
					updateParameters();
					updateOutput();
					renderer.setPixelRatio( params.renderScale );
					onWindowResize();
					resetHistory();

				},
				scenario( name ) {

					activeScenario = name;
					rods.visible = name === 'thin-objects';
					emitter.visible = name === 'bright-emitter';
					moving.visible = name === 'moving-reflected-object';
					camera.position.set( 3, 2, 3 );
					controls.target.set( 0, 0, 0 );
					if ( name === 'grazing-floor' ) {

						camera.position.set( 3, - 0.45, 3 );
						controls.target.set( 0, - 0.6, 0 );

					}
					resetHistory();

				},
				step( progress = 0 ) {

					if ( activeScenario === 'camera-translation' ) camera.position.set( 3 + progress * 0.75, 2, 3 );
					if ( activeScenario === 'screen-edge-exit' ) {

						camera.position.set( 3, 2, 3 );
						controls.target.set( progress * 1.8, 0, 0 );

					}
					moving.position.set( - 0.7 + progress * 1.4, - 0.25, 0.6 );
					moving.rotation.y = progress * 1.5;
					// The renderer's RAF loop is stopped: advance frame nodes exactly
					// once per completed test frame, without a display-rate limit.
					renderer._nodes.nodeFrame.update();
					renderer.info.reset();
					animate();
					return renderer.info.render.calls;

				},
				flush: () => renderer.backend.device.queue.onSubmittedWorkDone()
			};
` );
if ( source === html ) throw new Error( 'Example animation entry point changed; update the quality harness.' );
await mkdir( output, { recursive: true } );
const server = createServer( { root } );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const origin = `http://127.0.0.1:${ server.address().port }`;
const browser = await puppeteer.launch( {
	headless: true,
	...( chrome ? { executablePath: chrome } : {} ),
	args: [ '--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit' ],
	defaultViewport: { width, height, deviceScaleFactor: 1 }
} );
const report = { baseline, width, height, exposure: 1, effectScale: 1, renderScale: 1, mirrorBias: 0, comparisonRays: 2, comparisonSteps: 128, antialiasing: 'TRAA', source: 'Current frame', scenarios: [], checks: [] };
const errors = new Set();

async function newPage( useOriginal = false ) {

	const page = await browser.newPage();
	page.on( 'pageerror', error => errors.add( error.message ) );
	page.on( 'console', message => {

		if ( message.type() === 'error' && ! message.text().includes( '404' ) ) errors.add( message.text() );

	} );
	await page.setRequestInterception( true );
	page.on( 'request', request => {

		const url = new URL( request.url() );
		if ( url.pathname.endsWith( '/webgpu_postprocessing_ssr.html' ) ) return request.respond( { status: 200, contentType: 'text/html', body: source } );
		if ( useOriginal && url.pathname.endsWith( '/CoverageDenoiseNode.js' ) ) return request.respond( { status: 200, contentType: 'text/javascript', body: originalDenoiser } );
		if ( url.pathname.endsWith( '/favicon.ico' ) ) return request.respond( { status: 204 } );
		request.continue();

	} );
	await page.goto( origin );
	const adapter = await page.evaluate( async () => {

		const adapter = await navigator.gpu?.requestAdapter();
		if ( ! adapter ) return null;
		const info = adapter.info;
		return Object.fromEntries( [ 'vendor', 'architecture', 'device', 'description', 'isFallbackAdapter' ].map( key => [ key, info[ key ] ] ) );

	} );
	if ( ! adapter || adapter.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( adapter ) ) ) throw new Error( `Hardware WebGPU adapter required: ${ JSON.stringify( adapter ) }` );
	report.adapter = adapter;
	await page.goto( `${ origin }/examples/webgpu_postprocessing_ssr.html`, { waitUntil: 'networkidle0' } );
	await page.waitForFunction( () => window.__ssrQuality?.ready(), { timeout: 60000 } );
	await page.evaluate( () => {

		for ( const element of document.querySelectorAll( 'body > :not(canvas)' ) ) element.style.display = 'none';

	} );
	return page;

}

async function frames( page, count, moving = false, finalPose = false ) {

	return page.evaluate( async ( { count, moving, finalPose } ) => {

		const test = window.__ssrQuality;
		let calls;
		for ( let i = 0; i < count; i ++ ) {

			calls = test.step( finalPose ? 1 : moving ? i / Math.max( count - 1, 1 ) : 0 );
			await test.flush();

		}

		return calls;

	}, { count, moving, finalPose } );

}

async function snapshot( page, filename ) {

	const data = await page.screenshot( { path: path.join( output, filename ) } );
	return PNG.sync.read( Buffer.from( data ) );

}

function rmse( a, b ) {

	let sum = 0;
	for ( let i = 0; i < a.data.length; i ++ ) {

		if ( i % 4 !== 3 ) sum += ( ( a.data[ i ] - b.data[ i ] ) / 255 ) ** 2;

	}

	return Math.sqrt( sum / ( a.width * a.height * 3 ) );

}

try {

	const results = new Map();
	const profiles = [ ...( baseline ? [ 'original-filter' ] : [] ), 'upgraded-filter', 'high-sample-reference' ];
	for ( const profile of profiles ) {

		const page = await newPage( profile === 'original-filter' );
		await page.evaluate( profile => window.__ssrQuality.configure( {
			output: 'Beauty', reflectionSource: 'Current frame', antialiasing: 'TRAA', renderScale: 1, effectScale: 1,
			raysPerPixel: profile === 'high-sample-reference' ? 16 : 2,
			traceSteps: profile === 'high-sample-reference' ? 512 : 128,
			radius: profile === 'high-sample-reference' ? 0 : 1.5, maxFrames: 32
		} ), profile );
		for ( const scenario of scenarios ) {

			await page.evaluate( scenario => window.__ssrQuality.scenario( scenario ), scenario );
			const moving = [ 'camera-translation', 'moving-reflected-object', 'screen-edge-exit' ].includes( scenario );
			await frames( page, 8, false, profile === 'high-sample-reference' && moving );
			await frames( page, profile === 'high-sample-reference' ? 48 : 24, moving, profile === 'high-sample-reference' && moving );
			results.set( `${ profile }/${ scenario }`, await snapshot( page, `${ profile }-${ scenario }.png` ) );
			console.log( `${ profile }: ${ scenario } completed` );
			if ( errors.size ) throw new Error( [ ...errors ].join( '\n' ) );

		}

		if ( profile === 'upgraded-filter' ) {

			// Exercise switches and history restarts independently of quality metrics.
			for ( const source of [ 'Current frame', 'Previous frame' ] ) {

				for ( const aa of [ 'SMAA', 'TRAA' ] ) {

					for ( const scale of [ 1, 0.5, 0.25 ] ) {

						await page.evaluate( ( { source, aa, scale } ) => window.__ssrQuality.configure( { output: 'Beauty', reflectionSource: source, antialiasing: aa, effectScale: scale, renderScale: 1 } ), { source, aa, scale } );
						await frames( page, 4 );
						report.checks.push( { source, aa, scale } );

					}

				}

			}

			for ( const output of [ 'SSR', 'SSR Filtered', 'SSR Coverage', 'SSR Coverage Filtered', 'Indirect Specular', 'PMREM only', 'Beauty' ] ) {

				await page.evaluate( output => window.__ssrQuality.configure( { output, reflectionSource: 'Previous frame', effectScale: 1 } ), output );
				await frames( page, 4 );
				report.checks.push( { previousFrameOutput: output } );

			}

			for ( const source of [ 'Current frame', 'Previous frame' ] ) {

				await page.evaluate( source => {

					window.__ssrQuality.configure( { output: 'Beauty', reflectionSource: source, antialiasing: 'TRAA', effectScale: 1, renderScale: 1, raysPerPixel: 2, traceSteps: 128, maxFrames: 16 } );
					window.__ssrQuality.scenario( 'static' );

				}, source );
				const drawCalls = await frames( page, 16 );
				const benchmark = await page.evaluate( async () => {

					const test = window.__ssrQuality;
					await test.flush();
					let completedFrames = 0;
					const start = performance.now();
					while ( performance.now() - start < 5000 ) {

						for ( let i = 0; i < 4; i ++ ) test.step();
						await test.flush();
						completedFrames += 4;

					}

					await test.flush();
					return { completedFrames, seconds: ( performance.now() - start ) / 1000 };

				} );
				report.checks.push( { source, drawCalls, ...benchmark, fps: benchmark.completedFrames / benchmark.seconds } );
				console.log( `${ source }: ${ benchmark.completedFrames } GPU-completed frames in ${ benchmark.seconds.toFixed( 2 ) } seconds` );

			}

		}

		await page.close();

	}

	for ( const scenario of scenarios ) {

		const reference = results.get( `high-sample-reference/${ scenario }` );
		const current = results.get( `upgraded-filter/${ scenario }` );
		const original = results.get( `original-filter/${ scenario }` );
		report.scenarios.push( { scenario, upgradedRMSE: rmse( current, reference ), ...( original ? { originalRMSE: rmse( original, reference ) } : {} ) } );

	}

	report.errors = [ ...errors ];
	report.reference = 'Converged high-sample screen-space estimate at the same final pose; not ground-truth ray tracing. RMSE measures displayed RGB after identical tone mapping.';
	await writeFile( path.join( output, 'report.json' ), JSON.stringify( report, null, 2 ) );
	console.log( JSON.stringify( report, null, 2 ) );
	if ( errors.size ) throw new Error( [ ...errors ].join( '\n' ) );

} finally {

	await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

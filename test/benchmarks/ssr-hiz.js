// Separate Hi-Z construction and SSR-only GPU profiles on real examples.
// Cached scene inputs; blur, denoisers and scene rendering are excluded. Run from any cwd:
// PUPPETEER_EXECUTABLE_PATH=/path/to/Chrome node test/benchmarks/ssr-hiz.js
// Options: --example=mirror|stochastic|animation --width=1280 --height=720
//          --seconds=5 --adaptive-steps=512 --linear-steps=64 --max-distance=20
//          --candidates=1,2,3,4 --binary-refine=true --max-level=6 --glossy-min-level=1
//          --roughness-threshold=1 (default 1 traces every surface, keeping runs comparable)
// A fairer linear reference (one step per pixel, same thickness rule as adaptive):
//          --binary-refine=false --linear-steps=1024
// Vsync/frame caps are disabled. Each completed GPU batch is counted only after
// queue.onSubmittedWorkDone(). There are no per-frame timing queries.
import puppeteer from 'puppeteer';
import { fileURLToPath } from 'node:url';
import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from '../../utils/server.js';

const option = ( key, fallback ) => process.argv.find( value => value.startsWith( `--${ key }=` ) )?.split( '=' )[ 1 ] ?? fallback;
const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const examples = option( 'example', 'mirror,stochastic,animation' ).split( ',' );
const width = Number( option( 'width', 1280 ) );
const height = Number( option( 'height', 720 ) );
const seconds = Number( option( 'seconds', 5 ) );
const adaptiveSteps = Number( option( 'adaptive-steps', 512 ) );
const linearSteps = Number( option( 'linear-steps', 64 ) );
const maxDistance = option( 'max-distance', null );
const binaryRefine = option( 'binary-refine', 'true' ) === 'true';
const maxLevel = Number( option( 'max-level', Infinity ) );
const candidates = option( 'candidates', '1,2,3,4' ).split( ',' ).map( Number );
const server = createServer( { root } );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
let browser;
const results = [];
try {

	browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--disable-frame-rate-limit', '--disable-gpu-vsync' ] } );
	for ( const example of examples ) {

		if ( ! [ 'mirror', 'stochastic', 'animation' ].includes( example ) ) throw new Error( 'Unknown SSR example.' );
		const page = await browser.newPage();
		await page.setViewport( { width, height, deviceScaleFactor: 1 } );
		const errors = [];
		page.on( 'pageerror', error => errors.push( error.message ) );
		page.on( 'console', message => {

			if ( /THREE\.TSL:.*(?:Error|error)|GPUValidationError/.test( message.text() ) ) errors.push( message.text() );

		} );
		const filename = `webgpu_postprocessing_ssr_${ example }.html`;
		let html = await readFile( `${ root }examples/${ filename }`, 'utf8' );
		html = html.replaceAll( '../build/three.webgpu.js', '../src/Three.WebGPU.js' ).replaceAll( '../build/three.tsl.js', '../src/Three.TSL.js' );
		html = html.replaceAll( 'renderer.setAnimationLoop( animate );', '' );
		const node = example === 'mirror' ? 'reflection' : 'ssrNode';
		const params = example === 'stochastic' ? 'params.ssr' : 'params';
		const apply = example === 'stochastic' ? 'updateTracing' : 'updateParameters';
		const expose = `window.__ssrBenchmark = { renderer, node: ${ node }, params: ${ params }, apply: ${ apply }, render: () => renderPipeline.render() };`;
		if ( example === 'stochastic' ) html = html.replace( 'updateOutputNode();\n', `updateOutputNode();\n${ expose }\n` );
		else html = html.replace( 'updateOutput();\n', `updateOutput();\n${ expose }\n` );
		await page.setRequestInterception( true );
		page.on( 'request', request => {

			if ( new URL( request.url() ).pathname === `/examples/${ filename }` ) request.respond( { contentType: 'text/html', body: html } );
			else request.continue();

		} );
		await page.goto( `http://127.0.0.1:${ server.address().port }/examples/${ filename }` );
		await page.waitForFunction( () => window.__ssrBenchmark !== undefined, { timeout: 120000 } );
		await page.waitForNetworkIdle( { idleTime: 500, timeout: 120000 } );
		const result = await page.evaluate( async ( settings ) => {

			const { renderer, node, params, apply, render } = window.__ssrBenchmark;
			if ( renderer.backend.isWebGPUBackend !== true ) throw new Error( 'Hardware WebGPU is required.' );
			const device = renderer.backend.device;
			const info = device.adapterInfo;
			const adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
			if ( adapter.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( adapter ) ) ) throw new Error( 'Software adapters are not allowed.' );
			device.pushErrorScope( 'validation' );
			// The stochastic example re-applies its own params; the others have none.
			node.glossyMinLevel = params.glossyMinLevel = settings.glossyMinLevel;
			node.roughnessThreshold.value = params.roughnessThreshold = settings.roughnessThreshold;
			params.quality = 1;
			params.binaryRefine = settings.binaryRefine;
			params.adaptiveSteps = settings.adaptiveSteps;
			if ( settings.maxDistance !== null ) params.maxDistance = Number( settings.maxDistance );
			node.maxSteps.value = settings.linearSteps;
			const { DataUtils, NodeUpdateType, QuadMesh } = await import( '/src/Three.WebGPU.js' );
			const sourcePass = node.depthNode.passNode;
			if ( sourcePass === undefined ) throw new Error( 'Unable to freeze the producing depth pass.' );
			const images = [];
			const modes = [];
			const advance = () => {

				renderer._nodes.nodeFrame.update();
				node._noiseIndex.value = 1;

			};

			// Initialize the full example once, then freeze its G-buffer. Profiling
			// below draws only SSR's reflection material, excluding blur/denoisers.
			params.adaptive = true;
			apply();
			if ( settings.maxLevel !== null ) node.hiZ.maxLevel = settings.maxLevel;
			advance();
			render();
			await device.queue.onSubmittedWorkDone();
			const hierarchy = node.hiZ;
			const oldUpdateType = sourcePass.updateBeforeType;
			sourcePass.updateBeforeType = NodeUpdateType.NONE;
			hierarchy.enabled = false;
			const quad = new QuadMesh( node._ssrMaterial );
			const drawTrace = () => {

				advance();
				renderer.setRenderTarget( node.getRenderTarget() );
				quad.render( renderer );

			};

			const batchSize = 16;
			const batch = async ( draw ) => {

				for ( let i = 0; i < batchSize; i ++ ) draw();
				await device.queue.onSubmittedWorkDone();

			};

			const measure = async ( name, draw ) => {

				draw();
				await device.queue.onSubmittedWorkDone();
				const warm = performance.now();
				do {

					await batch( draw );

				} while ( performance.now() - warm < 500 );

				let completed = 0;
				const start = performance.now();
				do {

					await batch( draw );
					completed += batchSize;

				} while ( performance.now() - start < settings.seconds * 1000 );

				const elapsed = performance.now() - start;
				const fps = completed / ( elapsed / 1000 );
				return { name, completed, seconds: elapsed / 1000, fps, impliedMs: 1000 / fps };

			};

			hierarchy.enabled = true;
			const creation = await measure( 'Hi-Z creation', () => {

				advance();
				hierarchy.updateBefore( { renderer } );

			} );
			hierarchy.enabled = false;
			const configurations = [ { name: 'Non-adaptive SSR', adaptive: false, maxCandidates: 1 } ];
			for ( const maxCandidates of settings.candidates ) {

				configurations.push( { name: `Adaptive SSR (${ maxCandidates } candidate${ maxCandidates > 1 ? 's' : '' })`, adaptive: true, maxCandidates } );

			}

			const readTrace = async () => {

				const pixels = await renderer.readRenderTargetPixelsAsync( node.getRenderTarget(), 0, 0, renderer.domElement.width, renderer.domElement.height );
				return pixels instanceof Uint16Array ? Float32Array.from( pixels, value => DataUtils.fromHalfFloat( value ) ) : pixels;

			};

			const compare = ( a, b ) => {

				let baselineHits = 0, adaptiveHits = 0, lostHits = 0, extraHits = 0, nonFinite = 0;
				for ( let i = 0; i < a.length; i += 4 ) {

					const ah = a[ i + 3 ] > 0 && a[ i + 3 ] < 1000;
					const bh = b[ i + 3 ] > 0 && b[ i + 3 ] < 1000;
					if ( ah ) baselineHits ++;
					if ( bh ) adaptiveHits ++;
					if ( ah && ! bh ) lostHits ++;
					if ( bh && ! ah ) extraHits ++;
					for ( let c = 0; c < 4; c ++ ) if ( ! Number.isFinite( a[ i + c ] ) || ! Number.isFinite( b[ i + c ] ) ) nonFinite ++;

				}

				return { baselineHits, adaptiveHits, lostHits, extraHits, disagreements: lostHits + extraHits, nonFinite };

			};

			for ( const config of configurations ) {

				await device.queue.onSubmittedWorkDone();
				params.adaptive = config.adaptive;
				params.maxCandidates = config.maxCandidates;
				apply();
				const measurement = { ...config, ...await measure( config.name, drawTrace ) };
				const image = await readTrace();
				if ( images.length === 0 ) images.push( image );
				else measurement.comparison = compare( images[ 0 ], image );
				if ( config.adaptive ) {

					// Diagnostic draws are outside the measured interval.
					node._traceDebug = true;
					node._buildSSRMaterial();
					drawTrace();
					const diagnostic = await readTrace();
					let tracedRays = 0, iterations = 0, candidateCount = 0, exhausted = 0;
					for ( let i = 0; i < diagnostic.length; i += 4 ) {

						if ( diagnostic[ i ] === 0 ) continue;
						tracedRays ++;
						iterations += diagnostic[ i ];
						candidateCount += diagnostic[ i + 1 ];
						if ( diagnostic[ i + 3 ] > 0 ) exhausted ++;

					}

					measurement.diagnostics = { tracedRays, averageIterations: iterations / tracedRays, averageCandidates: candidateCount / tracedRays, exhaustedRays: exhausted };
					node._traceDebug = false;
					node._buildSSRMaterial();

				}

				modes.push( measurement );

			}

			sourcePass.updateBeforeType = oldUpdateType;

			const validationError = await device.popErrorScope();
			if ( validationError ) throw new Error( validationError.message );
			return { adapter, width: renderer.domElement.width, height: renderer.domElement.height, maxDistance: params.maxDistance, linearSteps: settings.linearSteps, binaryRefine: settings.binaryRefine, adaptiveSteps: settings.adaptiveSteps, levels: hierarchy.levels.length, creation, modes, speedup: modes[ 1 ].fps / modes[ 0 ].fps };

		}, { seconds, adaptiveSteps, linearSteps, maxDistance, candidates, binaryRefine, roughnessThreshold: Number( option( 'roughness-threshold', 1 ) ), glossyMinLevel: Number( option( 'glossy-min-level', 0 ) ), maxLevel: Number.isFinite( maxLevel ) ? maxLevel : null } );
		if ( errors.length > 0 ) throw new Error( errors.join( '\n' ) );
		results.push( { example, ...result } );
		console.log( JSON.stringify( results.at( - 1 ), null, 2 ) );
		await page.close();

	}

	const output = option( 'output', '/tmp/three-ssr-hiz-benchmark.json' );
	await writeFile( output, JSON.stringify( results, null, 2 ) + '\n' );
	console.log( `Results: ${ output }` );

} finally {

	if ( browser !== undefined ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

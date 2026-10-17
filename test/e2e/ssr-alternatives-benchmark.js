// node test/e2e/ssr-alternatives-benchmark.js [artifact-directory]
// Whole-frame throughput, including opaque/beauty scene renders, tracing, history,
// reconstruction and final postprocessing. Requires a physical WebGPU adapter.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';

const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const artifacts = path.resolve( process.argv[ 2 ] || process.env.SWARM_ARTIFACTS || path.join( os.tmpdir(), 'three-ssr-alternatives' ) );
const launchFlags = [ '--disable-frame-rate-limit', '--disable-gpu-vsync' ];
const controls = { source: 1, resolution: 1, steps: 16, rays: 1, seed: 1, reach: 1, thickness: 0.03, clamp: 0 };
const report = {
	status: 'failed',
	measurement: 'GPU-completed whole frames; sequential eight-frame batches with queue completion before counting',
	viewport: { width: 960, height: 640, deviceScaleFactor: 1 },
	launchFlags,
	controls,
	warmupSeconds: 1,
	measurementSeconds: 5,
	runsPerCase: 3,
	batchSize: 8,
	unmatched: [ 'Sampler and sampling/step distributions differ', 'Power mode uses previous opaque radiance; stock GGX uses current beauty', 'Seed control applies to power mode; stock GGX retains its existing noise generator', 'Postprocessing is matched within each example; the two examples have different postprocessing' ],
	cases: [],
	errors: [],
	warnings: []
};
const server = createServer( { root } );
const physical = /nvidia|geforce|quadro|radeon|\bamd\b|intel|apple|qualcomm|adreno|mali|samsung|imagination|10de|1002|8086|106b/i;
const software = /swiftshader|llvmpipe|lavapipe|software|microsoft basic|\bwarp\b/i;
let browser;

try {

	await mkdir( artifacts, { recursive: true } );
	await new Promise( ( resolve, reject ) => {

		server.once( 'error', reject );
		server.listen( 0, '127.0.0.1', resolve );

	} );
	const origin = `http://127.0.0.1:${ server.address().port }`;
	browser = await puppeteer.launch( { headless: true, args: launchFlags } );
	report.browser = await browser.version();

	for ( const example of [ 'webgpu_postprocessing_ssr', 'webgpu_postprocessing_ssr_denoise' ] ) {

		for ( const denoise of [ 0, 1 ] ) {

			for ( const mode of [ 'ggx', 'power' ] ) {

				const page = await browser.newPage();
				try {

					await page.setViewport( report.viewport );
					await page.setRequestInterception( true );
					page.on( 'request', request => {

						if ( new URL( request.url() ).pathname === '/favicon.ico' ) request.respond( { status: 204 } );
						else request.continue();

					} );
					const query = new URLSearchParams( { ...controls, mode, denoise } );
					const url = `${ origin }/examples/${ example }.html?${ query }`;
					const identify = text => ( { example, mode, denoise, text } );
					page.on( 'pageerror', error => report.errors.push( identify( String( error ) ) ) );
					page.on( 'console', message => {

						if ( message.type() === 'error' ) report.errors.push( identify( message.text() ) );
						else if ( message.type() === 'warn' ) report.warnings.push( identify( message.text() ) );

					} );
					await page.goto( url, { waitUntil: 'load', timeout: 120000 } );
					await page.waitForFunction( () => {

						const comparison = window.ssrComparison;
						return comparison?.renderPipeline && comparison.scene.children.some( child => child.isGroup && child.children.length > 0 );

					}, { timeout: 120000, polling: 100 } );
					const hardware = await page.evaluate( async () => {

						const comparison = window.ssrComparison;
						comparison.renderer.setAnimationLoop( null );
						comparison.controls.enableDamping = false;
						comparison.controls.enabled = false;
						comparison.controls.update();
						const adapter = await navigator.gpu?.requestAdapter();
						if ( ! adapter ) throw new Error( 'No hardware WebGPU adapter.' );
						const adapterInfo = adapter.info;
						const device = comparison.renderer.backend.device;
						const info = device.adapterInfo || adapterInfo;
						const canvas = document.createElement( 'canvas' );
						const gl = canvas.getContext( 'webgl2' );
						const extension = gl?.getExtension( 'WEBGL_debug_renderer_info' );
						const webglRenderer = extension ? gl.getParameter( extension.UNMASKED_RENDERER_WEBGL ) : '';
						gl?.getExtension( 'WEBGL_lose_context' )?.loseContext();
						window.ssrBenchmarkErrors = [];
						device.addEventListener( 'uncapturederror', event => window.ssrBenchmarkErrors.push( event.error.message ) );
						await device.queue.onSubmittedWorkDone();
						return {
							backend: comparison.renderer.backend.isWebGPUBackend ? 'WebGPU' : 'other',
							adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter ?? adapterInfo.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false },
							adapterSource: device.adapterInfo ? 'renderer GPUDevice.adapterInfo' : 'navigator.gpu.requestAdapter().info',
							webglRenderer
						};

					} );
					assert.equal( hardware.backend, 'WebGPU', 'The actual renderer must use WebGPU' );
					assert.equal( hardware.adapter.isFallbackAdapter, false, 'Fallback adapters are forbidden' );
					assert.ok( physical.test( JSON.stringify( hardware.adapter ) ) && ! software.test( JSON.stringify( hardware ) ), 'A physical WebGPU adapter is required' );
					assert.ok( physical.test( hardware.webglRenderer ), 'A physical WebGL renderer is required' );
					const result = { example, mode, denoise: Boolean( denoise ), url, hardware, runs: [] };
					report.cases.push( result );
					for ( let run = 0; run < report.runsPerCase; run ++ ) {

						const measurement = await page.evaluate( async ( { batchSize, warmupSeconds, measurementSeconds } ) => {

							const comparison = window.ssrComparison;
							const renderer = comparison.renderer;
							const queue = renderer.backend.device.queue;
							const counter = () => comparison.node._frame?.value ?? comparison.node._noiseIndex?.value;
							const batch = async () => {

								for ( let frame = 0; frame < batchSize; frame ++ ) {

									// Manual rendering has no animation loop to advance NodeFrame.
									renderer._nodes.nodeFrame.update();
									comparison.renderPipeline.render();

								}

								await queue.onSubmittedWorkDone();

							};

							await queue.onSubmittedWorkDone();
							comparison.resetHistory();
							const warmupStart = performance.now();
							while ( performance.now() - warmupStart < warmupSeconds * 1000 ) await batch();
							await queue.onSubmittedWorkDone();
							const counterBefore = counter();
							let counterAfterFirstBatch;
							let completedFrames = 0;
							const start = performance.now();
							do {

								await batch();
								// Frames are counted only after their GPU queue has completed.
								completedFrames += batchSize;
								if ( counterAfterFirstBatch === undefined ) counterAfterFirstBatch = counter();

							} while ( performance.now() - start < measurementSeconds * 1000 );

							await queue.onSubmittedWorkDone();
							const elapsedSeconds = ( performance.now() - start ) / 1000;
							return {
								completedFrames, elapsedSeconds, framesPerSecond: completedFrames / elapsedSeconds,
								noiseCounter: { before: counterBefore, afterFirstBatch: counterAfterFirstBatch, after: counter() },
								actualControls: {
									drawingBuffer: [ renderer.domElement.width, renderer.domElement.height ],
									resolutionScale: comparison.node.resolutionScale,
									steps: comparison.node.steps?.value ?? comparison.node.quality.value * 64,
									rays: comparison.node.rayCount?.value ?? 1,
									reach: comparison.node.maxRayLength?.value ?? comparison.node.maxDistance.value,
									thickness: comparison.node.thickness.value,
									maxLuminance: comparison.node.maxLuminance.value,
									postprocessing: comparison.params.post ?? 'SMAA and renderer tone mapping'
								},
								gpuErrors: [ ...window.ssrBenchmarkErrors ]
							};

						}, report );
						result.runs.push( measurement );
						assert.ok( measurement.completedFrames > 0 && measurement.elapsedSeconds >= report.measurementSeconds );
						const period = mode === 'power' ? 4096 : 0x7fffffff;
						assert.equal( ( measurement.noiseCounter.afterFirstBatch - measurement.noiseCounter.before + period ) % period, report.batchSize, 'Every measured whole frame must execute SSR tracing' );
						assert.deepEqual( measurement.gpuErrors, [], 'GPU validation errors invalidate the benchmark' );
						assert.deepEqual( measurement.actualControls.drawingBuffer, [ report.viewport.width, report.viewport.height ] );
						assert.equal( measurement.actualControls.resolutionScale, controls.resolution );
						assert.equal( measurement.actualControls.steps, controls.steps );
						assert.equal( measurement.actualControls.rays, controls.rays );
						assert.equal( measurement.actualControls.reach, controls.reach );
						assert.equal( measurement.actualControls.thickness, controls.thickness );
						console.log( `${ example } ${ mode } denoise=${ denoise } run ${ run + 1 }: ${ measurement.framesPerSecond.toFixed( 2 ) } completed frames/s` );

					}

					const rates = result.runs.map( run => run.framesPerSecond );
					result.meanFramesPerSecond = rates.reduce( ( sum, rate ) => sum + rate, 0 ) / rates.length;
					result.sampleStandardDeviation = Math.sqrt( rates.reduce( ( sum, rate ) => sum + ( rate - result.meanFramesPerSecond ) ** 2, 0 ) / ( rates.length - 1 ) );

				} finally {

					await page.close();

				}

			}

		}

	}

	assert.deepEqual( report.errors, [], 'Browser errors invalidate the benchmark' );
	assert.deepEqual( report.warnings.filter( warning => /GPUValidationError|validation error|WebGPU.*error/i.test( warning.text ) ), [], 'Validation warnings invalidate the benchmark' );
	report.status = 'passed';

} catch ( error ) {

	report.error = String( error.stack || error );
	console.error( report.error );
	process.exitCode = 1;

} finally {

	const cleanup = await Promise.allSettled( [
		browser?.close(),
		server.listening ? new Promise( ( resolve, reject ) => server.close( error => error ? reject( error ) : resolve() ) ) : Promise.resolve()
	] );
	for ( const result of cleanup ) {

		if ( result.status === 'rejected' ) {

			report.status = 'failed';
			report.errors.push( { cleanup: String( result.reason ) } );
			process.exitCode = 1;

		}

	}

	await mkdir( artifacts, { recursive: true } );
	const reportPath = path.join( artifacts, 'ssr-alternatives-benchmark.json' );
	await writeFile( reportPath, JSON.stringify( report, null, 2 ) + '\n' );
	console.log( `SSR whole-frame throughput report: ${ reportPath }` );

}

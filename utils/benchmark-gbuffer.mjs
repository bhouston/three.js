/* global console, process, GPUAdapter, URL, window, navigator, setTimeout, performance */
/**
 * Sustained whole-frame WebGPU throughput, with a GPU fence per bounded batch.
 * node utils/benchmark-gbuffer.mjs --out /path/to/artifacts
 * CHROME_PATH selects an existing browser. No software adapter is accepted.
 */
import puppeteer from 'puppeteer';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from './server.js';

const { values } = parseArgs( {
	options: {
		out: { type: 'string', default: 'gbuffer-results' },
		root: { type: 'string', default: path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' ) },
		smoke: { type: 'boolean', default: false },
		warmup: { type: 'string', default: '2' },
		seconds: { type: 'string', default: '5' },
		repetitions: { type: 'string', default: '3' },
		batch: { type: 'string', default: '8' },
		width: { type: 'string', default: '1280' },
		height: { type: 'string', default: '720' },
		models: { type: 'string' },
		help: { type: 'boolean', default: false }
	}
} );

if ( values.help ) {

	console.log( 'node utils/benchmark-gbuffer.mjs [--smoke] [--out directory] [--root repo] [--models comma-separated-root-relative-paths] [--seconds 5] [--warmup 2] [--repetitions 3] [--batch 8] [--width 1280] [--height 720]\nUses CHROME_PATH or the installed Puppeteer browser. Headful hardware WebGPU only.' );
	process.exit( 0 );

}

const numeric = name => {

	const value = Number( values[ name ] );
	if ( ! Number.isFinite( value ) || value <= 0 ) throw new Error( `Invalid --${ name }` );
	return value;

};
const options = {
	warmupMS: numeric( 'warmup' ) * 1000,
	durationMS: numeric( 'seconds' ) * 1000,
	repetitions: numeric( 'repetitions' ),
	batchSize: numeric( 'batch' ),
	width: numeric( 'width' ),
	height: numeric( 'height' )
};
for ( const key of [ 'repetitions', 'batchSize', 'width', 'height' ] ) {

	if ( ! Number.isInteger( options[ key ] ) ) throw new Error( `${ key } must be an integer` );

}
if ( options.batchSize > 32 ) throw new Error( 'Batch size must be at most 32 to bound in-flight frames' );

const out = path.resolve( values.out );
await mkdir( out, { recursive: true } );
const report = {
	started: new Date().toISOString(),
	options,
	smoke: values.smoke,
	method: 'Sequential configurations; warmup then timed wall interval; synchronous whole-frame renders in bounded batches; one GPU completion fence per batch, never per frame; count only fully fenced frames; final queue drain included in elapsed interval. No requestAnimationFrame in benchmark mode. Inspector CPU work and batch fence overhead are included.',
	limitations: [ 'Batch fences bound queued work and may reduce throughput relative to a deeper queue.', 'Configuration subtraction includes CPU scheduling, resource transitions and changed shader costs; it is not an isolated GPU stage timer.', 'No tracing or signal denoising performance is measured.' ],
	errors: [],
	results: [],
	smokeChecks: []
};
const launchArgs = [
	'--disable-frame-rate-limit',
	'--disable-gpu-vsync',
	'--disable-background-timer-throttling',
	'--disable-renderer-backgrounding',
	'--disable-backgrounding-occluded-windows'
];
const models = values.models ? values.models.split( ',' ) : [
	'examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf',
	'examples/models/gltf/LittlestTokyo.glb'
];
const matched = { mode: 'deferred', mobile: false, passMode: 'bypass', prepass: false, shadows: true, ao: false, traa: true, sharpen: false, resolution: 1, playing: false };
const configurations = [
	{ name: 'forward', params: { ...matched, mode: 'forward' } },
	{ name: 'mobile', params: { ...matched, mobile: true } },
	{ name: 'deferred-bypass', params: { ...matched } },
	{ name: 'deferred-materialized', params: { ...matched, passMode: 'materialized' } },
	{ name: 'deferred-prepass', params: { ...matched, prepass: true } },
	{ name: 'deferred-ao', params: { ...matched, ao: true } },
	{ name: 'deferred-no-traa', params: { ...matched, traa: false } }
];
const server = createServer( { root: path.resolve( values.root ) } );
let browser;

try {

	await new Promise( ( resolve, reject ) => {

		server.once( 'error', reject );
		server.listen( 0, '127.0.0.1', resolve );

	} );
	const executablePath = process.env.CHROME_PATH || puppeteer.executablePath();
	report.browser = { executablePath, launchArgs, headless: false };
	browser = await puppeteer.launch( { executablePath, headless: false, args: launchArgs, defaultViewport: { width: options.width, height: options.height, deviceScaleFactor: 1 }, protocolTimeout: 0 } );
	report.browser.version = await browser.version();
	const browserCDP = await browser.target().createCDPSession();
	report.browser.systemGPU = ( await browserCDP.send( 'SystemInfo.getInfo' ) ).gpu;
	const page = await browser.newPage();
	page.setDefaultTimeout( 120000 );
	page.on( 'pageerror', error => report.errors.push( { type: 'pageerror', message: error.message } ) );
	page.on( 'console', message => {

		if ( message.type() === 'error' ) report.errors.push( { type: 'console', message: message.text() } );

	} );
	await page.evaluateOnNewDocument( () => {

		globalThis.__gbufferGPUErrors = [];
		const original = GPUAdapter.prototype.requestDevice;
		GPUAdapter.prototype.requestDevice = async function ( ...args ) {

			const device = await original.apply( this, args );
			device.addEventListener( 'uncapturederror', event => globalThis.__gbufferGPUErrors.push( event.error.message ) );
			device.lost.then( info => {

				if ( info.reason !== 'destroyed' ) globalThis.__gbufferGPUErrors.push( `Device lost: ${ info.message }` );

			} );
			return device;

		};

	} );
	const url = new URL( `http://127.0.0.1:${ server.address().port }/examples/webgpu_gbuffer_performance.html` );
	url.searchParams.set( 'benchmark', '1' );
	url.searchParams.set( 'model', models[ 0 ] );
	report.url = url.href;
	await page.goto( url.href, { waitUntil: 'networkidle0' } );
	await page.waitForFunction( () => !! window.gbuffer );
	await page.evaluate( async () => await window.gbuffer.ready );
	report.adapter = await page.evaluate( async () => {

		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance', forceFallbackAdapter: false } );
		if ( ! adapter ) throw new Error( 'No WebGPU adapter' );
		const info = adapter.info;
		const result = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter };
		if ( info.isFallbackAdapter || ! info.vendor || /swiftshader|llvmpipe|lavapipe|software|microsoft basic/i.test( JSON.stringify( result ) ) ) throw new Error( `Hardware adapter required: ${ JSON.stringify( result ) }` );
		return result;

	} );

	async function configure( params ) {

		await page.evaluate( async params => {

			await window.gbuffer.drain();
			await window.gbuffer.configure( params );
			await window.gbuffer.drain();

		}, params );
		const snapshot = await page.evaluate( () => window.gbuffer.snapshot() );
		if ( ! /webgpu/i.test( String( snapshot.backend ) ) ) throw new Error( `App must confirm active WebGPU backend: ${ snapshot.backend }` );
		if ( snapshot.adapter?.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software|microsoft basic/i.test( JSON.stringify( snapshot.adapter ) ) ) throw new Error( 'App selected a software adapter' );
		return snapshot;

	}

	async function sample( warmupMS, durationMS ) {

		return await page.evaluate( async ( { warmupMS, durationMS, batchSize } ) => {

			const app = window.gbuffer;
			const yieldTask = () => globalThis.scheduler?.yield ? globalThis.scheduler.yield() : new Promise( resolve => setTimeout( resolve, 0 ) );
			async function interval( duration ) {

				await app.drain();
				let completedFrames = 0;
				let batches = 0;
				const start = performance.now();
				do {

					for ( let i = 0; i < batchSize; i ++ ) app.renderFrame();
					await app.drain();
					completedFrames += batchSize;
					batches ++;
					await yieldTask();

				} while ( performance.now() - start < duration );
				await app.drain();
				const elapsedMS = performance.now() - start;
				return { completedFrames, batches, elapsedMS, framesPerSecond: completedFrames * 1000 / elapsedMS };

			}
			const warmup = await interval( warmupMS );
			const measured = await interval( durationMS );
			return { warmup, ...measured, snapshot: app.snapshot(), gpuErrors: [ ...globalThis.__gbufferGPUErrors ] };

		}, { warmupMS, durationMS, batchSize: options.batchSize } );

	}

	if ( values.smoke ) {

		const smokeCases = [
			...configurations.map( c => ( { name: c.name, params: { ...c.params, model: models[ 0 ] } } ) ),
			{ name: 'substantial-compressed', params: { ...matched, model: models[ 1 ] || models[ 0 ] } },
			{ name: 'skinned', params: { ...matched, model: 'examples/models/gltf/Soldier.glb', playing: true } },
			{ name: 'morph', params: { ...matched, model: 'examples/models/gltf/AnimatedMorphSphere/glTF/AnimatedMorphSphere.gltf', playing: true } },
			{ name: 'instanced', params: { ...matched, model: 'examples/models/gltf/DamagedHelmet/glTF-instancing/DamagedHelmetGpuInstancing.gltf' } },
			{ name: 'transmission', params: { ...matched, model: 'examples/models/gltf/DragonAttenuation.glb' } }
		];
		for ( const test of smokeCases ) {

			try {

				await configure( test.params );
				const result = await sample( 200, 400 );
				report.smokeChecks.push( { ...test, ...result } );
				console.log( `Smoke ${ test.name }: ${ result.completedFrames } GPU-completed frames` );

			} catch ( error ) {

				report.errors.push( { type: 'smoke', test: test.name, message: error.message } );
				console.error( `Smoke ${ test.name }: ${ error.message }` );

			}

		}
		await page.setViewport( { width: 641, height: 359, deviceScaleFactor: 1 } );
		await configure( { ...matched, model: models[ 0 ] } );
		const before = await page.evaluate( () => window.gbuffer.snapshot() );
		const motions = await page.evaluate( async () => {

			const app = window.gbuffer;
			if ( typeof app.moveCamera !== 'function' || typeof app.moveObject !== 'function' ) throw new Error( 'Smoke requires moveCamera() and moveObject() hooks' );
			await app.moveCamera();
			app.renderFrame();
			await app.drain();
			const camera = app.snapshot();
			await app.moveObject();
			app.renderFrame();
			await app.drain();
			return { camera, object: app.snapshot() };

		} );
		report.smokeChecks.push( { name: 'odd-resize-camera-object-motion', before, ...motions, result: await sample( 200, 400 ) } );
		await page.screenshot( { path: path.join( out, 'smoke-odd-resize.png' ) } );

	} else {

		for ( const model of models ) {

			for ( const configuration of configurations ) {

				const params = { ...configuration.params, model };
				await configure( params );
				const repetitions = [];
				for ( let repetition = 0; repetition < options.repetitions; repetition ++ ) {

					const result = await sample( options.warmupMS, options.durationMS );
					repetitions.push( result );
					console.log( `${ model } ${ configuration.name } ${ repetition + 1 }/${ options.repetitions }: ${ result.framesPerSecond.toFixed( 2 ) } GPU-completed frames/s (${ result.completedFrames } / ${ ( result.elapsedMS / 1000 ).toFixed( 3 ) }s)` );

				}
				const rates = repetitions.map( r => r.framesPerSecond ).sort( ( a, b ) => a - b );
				const mean = rates.reduce( ( a, b ) => a + b, 0 ) / rates.length;
				const middle = Math.floor( rates.length / 2 );
				const median = rates.length % 2 ? rates[ middle ] : ( rates[ middle - 1 ] + rates[ middle ] ) / 2;
				const standardDeviation = Math.sqrt( rates.reduce( ( sum, value ) => sum + ( value - mean ) ** 2, 0 ) / rates.length );
				report.results.push( { name: configuration.name, params, repetitions, framesPerSecond: { min: rates[ 0 ], max: rates.at( - 1 ), median, mean, standardDeviation } } );
				await writeFile( path.join( out, 'benchmark-results.json' ), JSON.stringify( report, null, 2 ) );

			}

		}

	}
	const gpuErrors = await page.evaluate( () => globalThis.__gbufferGPUErrors );
	for ( const message of gpuErrors ) report.errors.push( { type: 'gpu', message } );
	await page.evaluate( async () => await window.gbuffer.drain() );
	if ( report.errors.length ) process.exitCode = 1;

} catch ( error ) {

	report.errors.push( { type: 'fatal', message: error.message, stack: error.stack } );
	console.error( error );
	process.exitCode = 1;

} finally {

	if ( browser ) await browser.close();
	server.closeAllConnections();
	await new Promise( resolve => server.close( resolve ) );
	report.finished = new Date().toISOString();
	await writeFile( path.join( out, values.smoke ? 'smoke-results.json' : 'benchmark-results.json' ), JSON.stringify( report, null, 2 ) );
	console.log( `Evidence: ${ out }` );

}


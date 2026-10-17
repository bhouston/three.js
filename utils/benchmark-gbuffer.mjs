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

		const runtimeErrors = report.errors.filter( error => [ 'pageerror', 'console', 'gpu' ].includes( error.type ) );
		if ( runtimeErrors.length ) throw new Error( `Cannot measure an invalid renderer: ${ JSON.stringify( runtimeErrors ) }` );
		const result = await page.evaluate( async ( { warmupMS, durationMS, batchSize } ) => {

			const app = window.gbuffer;
			const healthySnapshot = () => {

				const snapshot = app.snapshot();
				const errors = [ ...globalThis.__gbufferGPUErrors, ...( snapshot.errors || [] ) ];
				if ( errors.length ) throw new Error( `Invalid GPU/app frame: ${ JSON.stringify( errors ) }` );
				return snapshot;

			};
			const yieldTask = () => globalThis.scheduler?.yield ? globalThis.scheduler.yield() : new Promise( resolve => setTimeout( resolve, 0 ) );
			async function interval( duration ) {

				await app.drain();
				const firstSnapshot = healthySnapshot();
				const sourceFrameBefore = firstSnapshot.history?.current?.frame ?? 0;
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
				const lastSnapshot = healthySnapshot();
				const sourceFrameAfter = lastSnapshot.history?.current?.frame;
				const sourceFrameDelta = sourceFrameAfter - sourceFrameBefore;
				if ( sourceFrameDelta !== completedFrames ) throw new Error( `Incomplete source publication: ${ sourceFrameDelta } source frames for ${ completedFrames } GPU-completed frames (${ sourceFrameBefore } to ${ sourceFrameAfter })` );
				return { completedFrames, batches, elapsedMS, framesPerSecond: completedFrames * 1000 / elapsedMS, sourceFrameBefore, sourceFrameAfter, sourceFrameDelta };

			}
			const warmup = await interval( warmupMS );
			const measured = await interval( durationMS );
			return { warmup, ...measured, snapshot: app.snapshot(), gpuErrors: [ ...globalThis.__gbufferGPUErrors ] };

		}, { warmupMS, durationMS, batchSize: options.batchSize } );
		const lateErrors = report.errors.filter( error => [ 'pageerror', 'console', 'gpu' ].includes( error.type ) );
		if ( result.gpuErrors.length || result.snapshot.errors?.length || lateErrors.length ) throw new Error( 'GPU/app errors invalidate this throughput interval' );
		return result;

	}

	if ( values.smoke ) {

		const smokeCases = [
			...configurations.map( c => ( { name: c.name, params: { ...c.params, model: models[ 0 ] } } ) ),
			{ name: 'substantial-compressed', params: { ...matched, model: models[ 1 ] || models[ 0 ] } },
			{ name: 'forward-substantial-transparency', params: { ...matched, mode: 'forward', model: models[ 1 ] || models[ 0 ] } },
			{ name: 'mobile-substantial-transparency', params: { ...matched, mobile: true, model: models[ 1 ] || models[ 0 ] } },
			{ name: 'opaque-extension-fallback', params: { ...matched, model: 'examples/models/gltf/ClearcoatTest/ClearcoatTest.glb' } },
			{ name: 'shadows-off-resolution-sharpen', params: { ...matched, shadows: false, resolution: 0.75, sharpen: true, model: models[ 0 ] } },
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
		if ( before.width !== 641 || before.height !== 359 ) throw new Error( `Odd resize was not applied: ${ before.width } x ${ before.height }` );
		const readbacks = await page.evaluate( async () => {

			const app = window.gbuffer;
			if ( typeof app.readBuffer !== 'function' || typeof app.moveCamera !== 'function' || typeof app.moveObject !== 'function' ) throw new Error( 'Smoke requires readBuffer(), moveCamera() and moveObject() hooks' );
			const half = word => {

				const sign = word & 0x8000 ? - 1 : 1;
				const exponent = ( word >> 10 ) & 31;
				const fraction = word & 1023;
				if ( exponent === 31 ) return fraction ? NaN : sign * Infinity;
				return sign * ( exponent === 0 ? fraction * 2 ** - 24 : ( 1 + fraction / 1024 ) * 2 ** ( exponent - 15 ) );

			};
			const finite = ( data, label ) => {

				if ( ! data.length ) throw new Error( `${ label } readback is empty` );
				let min = Infinity, max = - Infinity, nonzero = 0;
				for ( const word of data ) {

					const value = half( word );
					if ( ! Number.isFinite( value ) ) throw new Error( `${ label } contains NaN/infinity (half word ${ word })` );
					min = Math.min( min, value );
					max = Math.max( max, value );
					if ( value !== 0 ) nonzero ++;

				}
				return { samples: data.length, min, max, nonzero };

			};
			const render = async count => {

				for ( let i = 0; i < count; i ++ ) app.renderFrame();
				await app.drain();

			};
			await render( 4 );
			const staticVelocity = [];
			for ( let frame = 0; frame < 4; frame ++ ) {

				await render( 1 );
				const velocity = finite( await app.readBuffer( 'surface', 3 ), 'static velocity' );
				staticVelocity.push( velocity );
				if ( velocity.nonzero ) throw new Error( `Static camera/geometry has ${ velocity.nonzero } nonzero velocity channels on jitter frame ${ frame }` );

			}
			const normals = finite( await app.readBuffer( 'surface', 1 ), 'oct normal' );
			if ( normals.min < - 1 || normals.max > 1 ) throw new Error( 'Octahedral normal channels are outside signed unit bounds' );
			const opaqueData = await app.readBuffer( 'opaque' );
			const opaque = finite( opaqueData, 'opaque color' );
			const beauty = finite( await app.readBuffer( 'beauty' ), 'beauty' );
			let rgbMin = Infinity, rgbMax = - Infinity;
			for ( let i = 0; i < opaqueData.length; i ++ ) if ( i % 4 !== 3 ) {

				const value = half( opaqueData[ i ] );
				rgbMin = Math.min( rgbMin, value );
				rgbMax = Math.max( rgbMax, value );

			}
			if ( rgbMax - rgbMin < 0.01 ) throw new Error( 'Opaque resolve is flat or empty' );
			await app.moveObject();
			await render( 1 );
			const movingVelocity = finite( await app.readBuffer( 'surface', 3 ), 'object velocity' );
			if ( ! movingVelocity.nonzero ) throw new Error( 'Moving object produced no surface velocity' );
			const object = app.snapshot();
			await app.moveCamera();
			await render( 2 );
			const camera = app.snapshot();
			const cutBeauty = finite( await app.readBuffer( 'beauty' ), 'camera-cut beauty' );
			const errors = [ ...globalThis.__gbufferGPUErrors, ...( camera.errors || [] ) ];
			if ( errors.length ) throw new Error( `Readback errors: ${ JSON.stringify( errors ) }` );
			return { staticVelocity, movingVelocity, normals, opaque: { ...opaque, rgbMin, rgbMax }, beauty, cutBeauty, camera, object };

		} );
		report.smokeChecks.push( { name: 'odd-resize-finite-color-and-motion', before, ...readbacks, result: await sample( 200, 400 ) } );
		await configure( { ...matched, model: 'examples/models/gltf/LittlestTokyo.glb', traa: false, prepass: false } );
		const prepassComparison = await page.evaluate( async () => {

			const app = window.gbuffer;
			const snapshots = [];
			const capture = async prepass => {

				await app.configure( { prepass } );
				for ( let i = 0; i < 3; i ++ ) app.renderFrame();
				await app.drain();
				snapshots.push( app.snapshot() );
				return await app.readBuffer( 'beauty' );

			};
			const off = await capture( false );
			const on = await capture( true );
			const restored = await capture( false );
			const half = word => {

				const sign = word & 0x8000 ? - 1 : 1;
				const exponent = ( word >> 10 ) & 31;
				const fraction = word & 1023;
				if ( exponent === 31 ) return fraction ? NaN : sign * Infinity;
				return sign * ( exponent === 0 ? fraction * 2 ** - 24 : ( 1 + fraction / 1024 ) * 2 ** ( exponent - 15 ) );

			};
			const compare = ( a, b, label ) => {

				if ( ! a.length || a.length !== b.length ) throw new Error( `${ label }: readback dimensions differ` );
				let maxAbsoluteError = 0, totalAbsoluteError = 0, changedChannels = 0, visibleChannels = 0;
				for ( let i = 0; i < a.length; i ++ ) if ( i % 4 !== 3 ) {

					const av = half( a[ i ] ), bv = half( b[ i ] );
					if ( ! Number.isFinite( av ) || ! Number.isFinite( bv ) ) throw new Error( `${ label }: NaN/infinity` );
					if ( av > 0.01 ) visibleChannels ++;
					const difference = Math.abs( av - bv );
					maxAbsoluteError = Math.max( maxAbsoluteError, difference );
					totalAbsoluteError += difference;
					if ( difference > Math.max( 0.002, Math.abs( av ) * 0.005 ) ) changedChannels ++;

				}
				const rgbChannels = a.length * 3 / 4;
				const changedFraction = changedChannels / rgbChannels;
				if ( ! visibleChannels ) throw new Error( `${ label }: empty beauty` );
				if ( changedFraction > 0.001 ) throw new Error( `${ label }: ${ ( changedFraction * 100 ).toFixed( 3 ) }% channels differ beyond half-float tolerance; prepass may corrupt alpha/depth or shared material state` );
				return { samples: rgbChannels, visibleChannels, changedChannels, changedFraction, maxAbsoluteError, meanAbsoluteError: totalAbsoluteError / rgbChannels };

			};
			const enabled = compare( off, on, 'prepass off/on alpha coverage' );
			const disabledAgain = compare( off, restored, 'prepass off/on/off material restoration' );
			const errors = [ ...globalThis.__gbufferGPUErrors, ...( app.snapshot().errors || [] ) ];
			if ( errors.length ) throw new Error( `Prepass comparison errors: ${ JSON.stringify( errors ) }` );
			return { model: 'examples/models/gltf/LittlestTokyo.glb', purpose: 'Contains MASK cutout materials and BLEND glass; static geometry, TRAA off, same lights/shadows/resolution; off/on/off checks coverage and shared material restoration.', enabled, disabledAgain, snapshots };

		} );
		report.smokeChecks.push( { name: 'prepass-alpha-coverage-and-material-restoration', ...prepassComparison } );
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


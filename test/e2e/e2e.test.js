import { afterAll, beforeAll, describe, test } from 'vitest';
import { chromium } from 'playwright';
import pLimit from 'p-limit';
import * as fs from 'fs/promises';
import * as os from 'os';
import { existsSync } from 'fs';
import { Image } from './image.js';
import { exceptionList } from './exception-list.js';
import { createServer } from '../../utils/server.js';

// Vitest-driven replacement for the old `node test/e2e/playwright.js` runner.
//
// A single Chromium process is shared by every example. Each example gets
// its own tab (browser context + page), opened right before it runs and
// closed right after, so no state (console listeners, routes, DOM, storage,
// in-flight requests) can leak between examples. `test.concurrent` lets
// vitest kick off all example tests immediately; a `p-limit` limiter bounds
// how many tabs actually run at once, which is what gives good load
// balancing (a tab that finishes a cheap example quickly picks up the next
// one, instead of a fixed static split of the file list).
//
// There's no lane-restart logic: since every example already gets a fresh
// context, a wedged example (e.g. a lost WebGPU device) is retried once in
// a brand-new tab rather than by tearing down and relaunching the browser.
//
// See test/e2e/README.md for usage (env vars for --webgpu / --make / an
// exact example list, and how this interacts with CI's own job matrix).

/* Configuration */

const pixelThreshold = 0.1; // threshold error in one pixel
const maxDifferentPixels = 0.1; // at most 0.1% different pixels

const idleTime = 0.4; // seconds - for how long there should be no network requests. Was 2s,
// tuned for real network latency; localhost round-trips are near-instant so a much shorter
// window is enough to catch a late-firing follow-up request.

const networkTimeout = 5; // 5 minutes, set to 0 to disable
const renderTimeout = 5; // 5 seconds, set to 0 to disable

const width = 400;
const height = 250;
const viewScale = 2;
const jpgQuality = 95;

const outputDir = 'test/e2e/output-screenshots';

// Number of tabs (contexts) allowed to run concurrently in the shared
// Chromium process. Override with E2E_WORKERS. Software rendering is
// CPU-bound, so going past the physical core count has shown no further
// speedup in testing (measured on an 8-core machine: 8 vs 16 was a wash) -
// default to the core count rather than oversubscribing.
const CONCURRENCY = Number( process.env.E2E_WORKERS ) || Math.max( 1, os.cpus().length );

const isMakeScreenshot = !! process.env.E2E_MAKE;
const isWebGPU = !! process.env.E2E_WEBGPU;
const exactList = ( process.env.E2E_ONLY || '' )
	.split( ',' )
	.map( s => s.trim() )
	.filter( Boolean );

/* Launch flags */

// `--enable-unsafe-swiftshader` is required from Chromium ~136 onward:
// software WebGL (SwiftShader) is no longer enabled by default in headless
// mode and silently fails ("Could not create a WebGL context") without it.
// The Vulkan/WebGPU flags rely on a system Vulkan driver (lavapipe via the
// `mesa-vulkan-drivers` package in CI; VK_DRIVER_FILES below points at it).
const launchArgs = [
	'--hide-scrollbars',
	'--enable-unsafe-webgpu',
	'--enable-features=Vulkan',
	'--disable-vulkan-surface',
	'--ignore-gpu-blocklist',
	'--disable-gpu-driver-bug-workarounds',
	'--disable-gpu-watchdog',
	'--use-gl=angle',
	'--use-angle=swiftshader-webgl',
	'--enable-unsafe-swiftshader',
	'--no-sandbox'
];

const viewport = { width: width * viewScale, height: height * viewScale };

// Points ANGLE/Vulkan at the lavapipe software ICD installed by
// `mesa-vulkan-drivers` in CI, which is what makes WebGPU work headlessly
// there. Only applied when that file actually exists - forcing it on a
// machine without it (e.g. local macOS/Windows dev, or a differently
// configured Linux box) makes the Vulkan loader fail hard enough to take
// down WebGL too, not just WebGPU.
const vulkanIcd = '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json';

const launchOptions = {
	headless: ( 'CI' in process.env || process.env.VISIBLE ) ? false : true,
	env: existsSync( vulkanIcd ) ? { ...process.env, VK_DRIVER_FILES: vulkanIcd } : process.env,
	args: launchArgs,
	timeout: 0
};

/* Shared server + build injection, set up once for the whole suite */

const server = createServer();
let port;

const buildInjection = ( code ) => code
	.replace( /Math\.random\(\) \* 0xffffffff/g, 'Math._random() * 0xffffffff' )
	// Disables WebGPU timestamp queries to prevent Inspector/Profiler from crashing in E2E software mode
	.replace( /this\.trackTimestamp\s*=\s*\(\s*parameters\.trackTimestamp\s*===\s*true\s*\);/g, 'Object.defineProperty(this, \'trackTimestamp\', { get: () => false, set: () => {} });' );

let cleanPage;
let injection;
let builds;

/* Shared browser process; every example runs in its own tab */

let browserServer;
let browser;
const limit = pLimit( CONCURRENCY );

// Opens a fresh tab (its own context + page) for a single example, so no
// state can leak between examples sharing the browser process.
async function openPage() {

	const context = await browser.newContext( { viewport } );
	const page = await context.newPage();
	await preparePage( page, injection, builds );

	return page;

}

async function closePage( page ) {

	try {

		await page.context().close();

	} catch ( e ) {}

}

function createNetworkTracker( page ) {

	let pending = 0;
	let lastActivity = Date.now();

	const onRequest = () => {

		pending ++;
		lastActivity = Date.now();

	};

	const onDone = () => {

		pending = Math.max( 0, pending - 1 );
		lastActivity = Date.now();

	};

	page.on( 'request', onRequest );
	page.on( 'requestfinished', onDone );
	page.on( 'requestfailed', onDone );

	return {

		// Waits until there have been no in-flight requests for `idleTime` ms.
		async waitForIdle( { idleTime = 500, timeout = 0 } = {} ) {

			const start = Date.now();

			while ( true ) {

				if ( pending === 0 && ( Date.now() - lastActivity ) >= idleTime ) return;

				if ( timeout && ( Date.now() - start ) > timeout ) {

					throw new Error( 'waitForNetworkIdle timeout exceeded' );

				}

				await new Promise( resolve => setTimeout( resolve, 100 ) );

			}

		}

	};

}

async function preparePage( page, injection, builds ) {

	await page.addInitScript( injection );

	page.networkTracker = createNetworkTracker( page );
	page.consoleErrors = [];
	page.consoleWarnings = [];

	const seenMessages = new Set();

	page.on( 'console', async msg => {

		const type = msg.type();
		const file = page.file;

		if ( file === undefined ) return;

		const args = await Promise.all( msg.args().map( async arg => {

			try {

				return await arg.evaluate( arg => arg instanceof Error ? arg.message : arg );

			} catch ( e ) {

				// Execution context might have been already destroyed
				return arg;

			}

		} ) );

		let text = args.join( ' ' );

		text = text.trim();
		if ( text === '' ) return;
		if ( text.includes( 'Timestamp tracking is disabled' ) ) return;
		// ANGLE's SwiftShader backend (SwANGLE) never implements
		// GL_KHR_parallel_shader_compile on any platform - it's not a flag or
		// config gap, so this warning is expected/unavoidable under the
		// software rendering this suite deliberately uses for reproducibility.
		if ( text.includes( 'KHR_parallel_shader_compile extension not supported' ) ) return;

		text = text.replace( /\[\.WebGL-(.+?)\] /g, '' );

		const key = file + ': ' + text;
		if ( seenMessages.has( key ) ) return;
		seenMessages.add( key );

		if ( type === 'warning' ) {

			console.log( `[Browser] ${key}` );
			page.consoleWarnings.push( text );

		} else if ( type === 'error' ) {

			page.consoleErrors.push( text );

		} else {

			console.log( `[Browser] ${key}` );

		}

	} );

	await page.route( '**/*', async ( route ) => {

		const url = route.request().url();

		for ( const build in builds ) {

			if ( url === `http://localhost:${ port }/build/${ build }` ) {

				await route.fulfill( {
					status: 200,
					contentType: 'application/javascript; charset=utf-8',
					body: builds[ build ]
				} );

				return;

			}

		}

		await route.continue();

	} );

}

async function renderAndScreenshot( page, file ) {

	page.file = file;
	page.consoleErrors = [];
	page.consoleWarnings = [];

	try {

		await page.goto( `http://localhost:${ port }/examples/${ file }.html`, {
			waitUntil: 'domcontentloaded',
			timeout: networkTimeout * 60000
		} );

		await page.networkTracker.waitForIdle( {
			idleTime: 500, // matches puppeteer's 'networkidle0' semantics
			timeout: networkTimeout * 60000
		} );

	} catch ( e ) {

		throw new Error( `Error happened while loading file ${ file }: ${ e }` );

	}

	try {

		await page.evaluate( cleanPage );

		await page.networkTracker.waitForIdle( {
			idleTime: idleTime * 1000,
			timeout: networkTimeout * 60000
		} );

		await page.evaluate( async ( { renderTimeout } ) => {

			// Wait for the main thread to actually go idle (worker-based decode,
			// shader compile, texture upload, etc. all show up as scheduled work)
			// instead of guessing a delay from downloaded byte count. Bounded by
			// a short timeout so an example that's never truly idle doesn't stall.
			await new Promise( resolve => {

				if ( 'requestIdleCallback' in window ) requestIdleCallback( resolve, { timeout: 500 } );
				else setTimeout( resolve, 50 );

			} );

			window._renderStarted = true;

			await new Promise( function ( resolve, reject ) {

				const renderStart = performance._now();

				const waitingLoop = setInterval( function () {

					const renderTimeoutExceeded = ( renderTimeout > 0 ) && ( performance._now() - renderStart > 1000 * renderTimeout );

					if ( renderTimeoutExceeded ) {

						clearInterval( waitingLoop );
						reject( 'Render timeout exceeded' );

					} else if ( window._renderFinished ) {

						clearInterval( waitingLoop );
						resolve();

					}

				}, 16 );

			} );

		}, { renderTimeout } );

	} catch ( e ) {

		if ( String( e ).includes( 'Render timeout exceeded' ) === false ) {

			throw new Error( `Error happened while rendering file ${ file }: ${ e }` );

		} // else: TODO fix this - some examples never resolve _renderFinished

	}

	const screenshot = ( await Image.read( await page.screenshot() ) ).scale( 1 / viewScale );

	if ( page.consoleErrors.length ) {

		throw new Error( `Console error(s) in file ${ file }:\n${ page.consoleErrors.join( '\n' ) }` );

	}

	return screenshot;

}

async function checkFile( file ) {

	const pageStart = performance.now();

	let page = await openPage();

	try {

		const screenshot = await renderAndScreenshot( page, file );

		const pageElapsed = ( performance.now() - pageStart ) / 1000;

		if ( isMakeScreenshot ) {

			await screenshot.write( `examples/screenshots/${ file }.jpg`, jpgQuality );
			return;

		}

		let expected;

		try {

			expected = await Image.read( `examples/screenshots/${ file }.jpg` );

		} catch ( e ) {

			await screenshot.write( `${ outputDir }/${ file }-actual.jpg`, jpgQuality );
			throw new Error( `Screenshot does not exist: ${ file }` );

		}

		const actual = screenshot.bitmap;
		const diff = screenshot.clone();

		let numDifferentPixels;

		try {

			numDifferentPixels = expected.compare( screenshot, diff, pixelThreshold );

		} catch ( e ) {

			await screenshot.write( `${ outputDir }/${ file }-actual.jpg`, jpgQuality );
			await expected.write( `${ outputDir }/${ file }-expected.jpg`, jpgQuality );
			throw new Error( `Image sizes do not match in file: ${ file }` );

		}

		const differentPixels = numDifferentPixels / ( actual.width * actual.height ) * 100;

		if ( differentPixels >= maxDifferentPixels ) {

			await screenshot.write( `${ outputDir }/${ file }-actual.jpg`, jpgQuality );
			await expected.write( `${ outputDir }/${ file }-expected.jpg`, jpgQuality );
			await diff.write( `${ outputDir }/${ file }-diff.jpg`, jpgQuality );
			throw new Error( `Diff wrong in ${ differentPixels.toFixed( 1 ) }% of pixels in file: ${ file } (${ pageElapsed.toFixed( 1 ) }s)` );

		}

	} catch ( e ) {

		// A lost WebGPU device (e.g. another process on the machine starving
		// the GPU) tends to cascade: every subsequent GPUValidationError on
		// that page follows from the original loss but usually doesn't repeat
		// the literal "Device Lost" message itself - only the first example to
		// observe it does. Matching the wider cascade signature too means a
		// wedged tab gets replaced by a fresh one on the very next attempt
		// instead of failing outright.
		const isDeviceLossOrCascade = /WebGPU Device Lost|GPUValidationError|is invalid due to a previous error/.test( String( e ) );

		if ( isDeviceLossOrCascade ) {

			console.warn( `${ e }` );
			console.warn( `Retrying ${ file } in a fresh tab after device loss...` );
			await closePage( page );
			page = await openPage();

			// Give the example a single retry in a brand-new tab before
			// failing outright - mirrors the old script's recovery.
			const screenshot = await renderAndScreenshot( page, file );

			if ( ! isMakeScreenshot ) {

				const expected = await Image.read( `examples/screenshots/${ file }.jpg` );
				const diff = screenshot.clone();
				const numDifferentPixels = expected.compare( screenshot, diff, pixelThreshold );
				const differentPixels = numDifferentPixels / ( screenshot.bitmap.width * screenshot.bitmap.height ) * 100;

				if ( differentPixels >= maxDifferentPixels ) {

					throw new Error( `Diff wrong in ${ differentPixels.toFixed( 1 ) }% of pixels in file: ${ file } (after device-lost retry)` );

				}

			} else {

				await screenshot.write( `examples/screenshots/${ file }.jpg`, jpgQuality );

			}

		} else {

			throw e;

		}

	} finally {

		await closePage( page );

	}

}

/* Discover which examples to test */

async function listFiles() {

	let files = ( await fs.readdir( 'examples' ) )
		.filter( s => s.slice( - 5 ) === '.html' && s !== 'index.html' )
		.map( s => s.slice( 0, s.length - 5 ) )
		.filter( f => exactList.length ? exactList.includes( f ) : ! exceptionList.includes( f ) );

	if ( exactList.length ) {

		for ( const file of exactList ) {

			if ( ! files.includes( file ) ) {

				console.log( `Warning! Unrecognised example name: ${ file }` );

			}

		}

	}

	if ( isWebGPU ) files = files.filter( f => f.includes( 'webgpu_' ) );

	// CI parallelism across job-matrix shards, e.g. CI=0..4 with numCIJobs=5.
	// Independent from CONCURRENCY, which parallelizes *within* one shard/machine.
	if ( 'CI' in process.env && process.env.CI !== '' && ! isNaN( Number( process.env.CI ) ) ) {

		const numCIJobs = Number( process.env.E2E_CI_JOBS ) || 5;
		const CI = Number( process.env.CI );

		files = files.slice(
			Math.floor( CI * files.length / numCIJobs ),
			Math.floor( ( CI + 1 ) * files.length / numCIJobs )
		);

	}

	return files;

}

/* Suite */

beforeAll( async () => {

	try {

		await fs.rm( outputDir, { recursive: true, force: true } );

	} catch ( e ) {}

	await fs.mkdir( outputDir, { recursive: true } );

	cleanPage = await fs.readFile( 'test/e2e/clean-page.js', 'utf8' );
	injection = await fs.readFile( 'test/e2e/deterministic-injection.js', 'utf8' );

	builds = {
		'three.core.js': buildInjection( await fs.readFile( 'build/three.core.js', 'utf8' ) ),
		'three.module.js': buildInjection( await fs.readFile( 'build/three.module.js', 'utf8' ) ),
		'three.webgpu.js': buildInjection( await fs.readFile( 'build/three.webgpu.js', 'utf8' ) )
	};

	port = await new Promise( ( resolve, reject ) => {

		server.once( 'error', reject );
		server.listen( 0, () => resolve( server.address().port ) );

	} );

	// launchServer (instead of chromium.launch()) gives us a handle on the
	// underlying browser process for a clean kill() in afterAll.
	browserServer = await chromium.launchServer( launchOptions );
	browser = await chromium.connect( browserServer.wsEndpoint() );

}, 5 * 60000 );

afterAll( async () => {

	try {

		await browserServer.kill();

	} catch ( e ) {}

	await new Promise( resolve => server.close( resolve ) );

}, 60000 );

const files = await listFiles();

console.log( `Testing ${ files.length } example(s) with up to ${ CONCURRENCY } concurrent tab(s)${ isMakeScreenshot ? ' (generating screenshots)' : '' }.` );

describe.concurrent( 'examples', () => {

	for ( const file of files ) {

		test( file, { timeout: 0 }, () => limit( () => checkFile( file ) ) );

	}

} );

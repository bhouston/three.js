import { chromium } from 'playwright';
import { Image } from './image.js';
import * as fs from 'fs/promises';
import { createServer } from '../../utils/server.js';

const server = createServer();

// If an example needs to be skipped, add it here.
// It is duplicated here (rather than imported) because puppeteer.js has
// top-level side effects (it starts an HTTP server and launches a browser as
// soon as it is loaded), so importing from it would not be safe.
const exceptionList = [

	// Take too long
	'webgpu_cubemap_mix', 				// 2 min
	'webgl_loader_texture_ultrahdr', 	// 1 min
	'webgl_marchingcubes', 				// 1 min
 	'webgl_materials_cubemap_dynamic', 	// 1 min
	'webgl_materials_displacementmap', 	// 1 min
	'webgl_materials_envmaps_hdr', 		// 1 min
	'webgpu_water', 					// 1 min

	// Requires HTML-in-Canvas API
	'webgl_materials_texture_html',
	'webgpu_materials_texture_html',

	// Black screen
	'webgpu_postprocessing_ao',
	'webgpu_postprocessing_dof',
	'webgpu_postprocessing_ssgi',
	'webgpu_postprocessing_ssgi_ballpool',
	'webgpu_postprocessing_sss',
	'webgpu_postprocessing_traa',
	'webgpu_tsl_vfx_linkedparticles',
	'webgpu_volume_lighting_traa',

	// Timming issues?
	'physics_rapier_instancing',
	'webgl_shadowmap',
	'webaudio_visualizer',
	'webgpu_compute_audio',
	'webgpu_compute_cloth',
	'webgpu_compute_particles_fluid',
	'webgpu_compute_rasterizer_ibl', // Rasterizer discrepancies
	'webgpu_compute_sort_bitonic',
	'webgpu_storage_buffer',
	'webgpu_tsl_editor',
	'webgpu_tsl_graph',
	'webxr_vr_video',
	'webgpu_tsl_transpiler',
	'webgpu_rendertarget_2d-array_3d',
	'webgpu_volume_fire',

	// Need more time to render
	'css3d_mixed',
	'webgl_loader_3dtiles',
	'webgl_loader_texture_lottie',
	'webgl_morphtargets_face',
	'webgl_renderer_pathtracer',
	'webgl_shadowmap_progressive',
	'webgpu_materials_matcap',
	'webgpu_morphtargets_face',
	'webgpu_shadowmap_progressive',
	'webgpu_postprocessing_ssr_denoise',

	// Video hangs the CI?
	'css3d_youtube',
	'webgpu_materials_video',
	'webgl_video_kinect',
	'webgl_video_panorama_equirectangular',

	// Timeout
	'webgl_test_memory2',

	// Webcam
	'webgl_materials_video_webcam',
	'webgl_morphtargets_webcam',

	// Sub-pixel coverage of thin high-contrast geometry edges differs across rasterizers #33817
	'webgpu_generator_city'

];

/* Configuration */

const port = 1234;
const pixelThreshold = 0.1; // threshold error in one pixel
const maxDifferentPixels = 0.1; // at most 0.1% different pixels

const idleTime = 2; // 2 seconds - for how long there should be no network requests
const parseTime = 1; // 1 second per megabyte

const networkTimeout = 5; // 5 minutes, set to 0 to disable
const renderTimeout = 5; // 5 seconds, set to 0 to disable
const numCIJobs = 5; // GitHub Actions run the script in 5 threads

const width = 400;
const height = 250;
const viewScale = 2;
const jpgQuality = 95;

const outputDir = 'test/e2e/output-screenshots';

console.red = msg => console.log( `\x1b[31m${msg}\x1b[39m` );
console.green = msg => console.log( `\x1b[32m${msg}\x1b[39m` );
console.yellow = msg => console.log( `\x1b[33m${msg}\x1b[39m` );

let browserServer;

/* Launch server */

server.listen( port, main );

process.on( 'SIGINT', async () => {

	console.log( '\nInterrupted, cleaning up...' );

	if ( browserServer ) {

		try {

			await browserServer.close();

		} catch ( e ) {}

	}

	server.close();
	process.exit( 1 );

} );

async function main() {

	/* Create output directory */

	try {

		await fs.rm( outputDir, { recursive: true, force: true } );

	} catch ( e ) {}

	try {

		await fs.mkdir( outputDir, { recursive: true } );

	} catch ( e ) {}

	/* Find files */

	let isMakeScreenshot = false;
	let isWebGPU = false;

	let argvIndex = 2;

	if ( process.argv[ argvIndex ] === '--webgpu' ) {

		isWebGPU = true;
		argvIndex ++;

	}

	if ( process.argv[ argvIndex ] === '--make' ) {

		isMakeScreenshot = true;
		argvIndex ++;

	}

	const exactList = process.argv.slice( argvIndex )
		.map( f => f.replace( '.html', '' ) );

	const isExactList = exactList.length !== 0;

	let files = ( await fs.readdir( 'examples' ) )
		.filter( s => s.slice( - 5 ) === '.html' && s !== 'index.html' )
		.map( s => s.slice( 0, s.length - 5 ) )
		.filter( f => isExactList ? exactList.includes( f ) : ! exceptionList.includes( f ) );

	if ( isExactList ) {

		for ( const file of exactList ) {

			if ( ! files.includes( file ) ) {

				console.log( `Warning! Unrecognised example name: ${ file }` );

			}

		}

	}

	if ( isWebGPU ) files = files.filter( f => f.includes( 'webgpu_' ) );

	/* CI parallelism */
	// note: if this port is switched to the native `@playwright/test` runner
	// in the future, this can be replaced by `--shard=x/y`.

	if ( 'CI' in process.env ) {

		const CI = parseInt( process.env.CI );

		files = files.slice(
			Math.floor( CI * files.length / numCIJobs ),
			Math.floor( ( CI + 1 ) * files.length / numCIJobs )
		);

	}

	/* Launch browser */

	const flags = [
		'--hide-scrollbars',
		'--enable-unsafe-webgpu',
		'--enable-features=Vulkan',
		'--disable-vulkan-surface',
		'--ignore-gpu-blocklist',
		'--disable-gpu-driver-bug-workarounds',
		'--disable-gpu-watchdog',
		'--no-sandbox'
	];

	const viewport = { width: width * viewScale, height: height * viewScale };

	const launchOptions = {
		headless: ( 'CI' in process.env || process.env.VISIBLE ) ? false : true,
		env: { ...process.env, VK_DRIVER_FILES: '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json' },
		args: flags,
		timeout: 0
	};

	/* Prepare injections */

	const buildInjection = ( code ) => code
		.replace( /Math\.random\(\) \* 0xffffffff/g, 'Math._random() * 0xffffffff' )
		// Disables WebGPU timestamp queries to prevent Inspector/Profiler from crashing in E2E software mode
		.replace( /this\.trackTimestamp\s*=\s*\(\s*parameters\.trackTimestamp\s*===\s*true\s*\);/g, 'Object.defineProperty(this, \'trackTimestamp\', { get: () => false, set: () => {} });' );

	const cleanPage = await fs.readFile( 'test/e2e/clean-page.js', 'utf8' );
	const injection = await fs.readFile( 'test/e2e/deterministic-injection.js', 'utf8' );

	const builds = {
		'three.core.js': buildInjection( await fs.readFile( 'build/three.core.js', 'utf8' ) ),
		'three.module.js': buildInjection( await fs.readFile( 'build/three.module.js', 'utf8' ) ),
		'three.webgpu.js': buildInjection( await fs.readFile( 'build/three.webgpu.js', 'utf8' ) )
	};

	/* Prepare page */

	const errorMessagesCache = [];

	const launchPage = async () => {

		// launchServer (instead of chromium.launch()) gives us a handle on the
		// underlying browser process so we can force-kill it if it wedges,
		// mirroring puppeteer.js's SIGKILL-based restart logic.
		browserServer = await chromium.launchServer( launchOptions );
		const browser = await chromium.connect( browserServer.wsEndpoint() );
		const context = await browser.newContext( { viewport } );
		const page = await context.newPage();
		await preparePage( page, injection, builds, errorMessagesCache );
		return page;

	};

	const ctx = {
		page: await launchPage(),
		async restart() {

			// Kill the whole Chrome process tree; browserServer.close() can hang
			// after a wedged GPU process, so use kill() which SIGKILLs it.
			try {

				await browserServer.kill();

			} catch ( e ) {}

			errorMessagesCache.length = 0;
			ctx.page = await launchPage();

		}
	};

	/* Loop for each file */

	const failedScreenshots = [];

	for ( const file of files ) {

		await checkFile( ctx, failedScreenshots, cleanPage, isMakeScreenshot, file );

	}

	/* Finish */

	failedScreenshots.sort();
	const list = failedScreenshots.join( ' ' );

	if ( isMakeScreenshot && failedScreenshots.length ) {

		console.red( 'List of failed screenshots: ' + list );
		console.red( `If you are sure that everything is correct, try to run "npm run make-screenshot ${ list }". If this does not help, add remaining screenshots to the exception list.` );
		console.red( `${ failedScreenshots.length } from ${ files.length } screenshots have not generated successfully.` );

	} else if ( isMakeScreenshot && ! failedScreenshots.length ) {

		console.green( `${ files.length } screenshots successfully generated.` );

	} else if ( failedScreenshots.length ) {

		console.red( 'List of failed screenshots: ' + list );
		console.red( `If you are sure that everything is correct, try to run "npm run make-screenshot ${ list }". If this does not help, add remaining screenshots to the exception list.` );
		console.red( `TEST FAILED! ${ failedScreenshots.length } from ${ files.length } screenshots have not rendered correctly.` );

	} else {

		console.green( `TEST PASSED! ${ files.length } screenshots rendered correctly.` );

	}

	setTimeout( close, 300, failedScreenshots.length );

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
		// This replicates puppeteer's page.waitForNetworkIdle({ idleTime, timeout }),
		// which Playwright does not provide directly (Playwright's own
		// 'networkidle' load state uses a fixed, non-configurable 500ms window).
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

async function preparePage( page, injection, builds, errorMessages ) {

	await page.addInitScript( injection );

	page.networkTracker = createNetworkTracker( page );

	page.on( 'console', async msg => {

		const type = msg.type();

		const file = page.file;

		if ( file === undefined ) {

			return;

		}

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

		text = file + ': ' + text.replace( /\[\.WebGL-(.+?)\] /g, '' );

		if ( errorMessages.includes( text ) ) {

			return;

		}

		errorMessages.push( text );

		if ( type === 'warning' ) {

			console.yellow( text );

		} else if ( type === 'error' ) {

			page.error = text;

		} else {

			console.log( `[Browser] ${text}` );

		}

	} );

	page.on( 'response', async ( response ) => {

		try {

			if ( response.status() === 200 ) {

				const buffer = await response.body();
				page.pageSize += buffer.length;

			}

		} catch ( e ) {}

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

async function checkFile( ctx, failedScreenshots, cleanPage, isMakeScreenshot, file ) {

	const page = ctx.page;
	const pageStart = performance.now();

	try {

		page.file = file;
		page.pageSize = 0;
		page.error = undefined;

		/* Load target page */

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

			/* Render page */

			await page.evaluate( cleanPage );

			await page.networkTracker.waitForIdle( {
				idleTime: idleTime * 1000,
				timeout: networkTimeout * 60000
			} );

			await page.evaluate( async ( { renderTimeout, parseTime } ) => {

				await new Promise( resolve => setTimeout( resolve, parseTime ) );

				/* Resolve render promise */

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

					}, 100 );

				} );

			}, { renderTimeout, parseTime: page.pageSize / 1024 / 1024 * parseTime * 1000 } );

		} catch ( e ) {

			if ( String( e ).includes( 'Render timeout exceeded' ) === false ) {

				throw new Error( `Error happened while rendering file ${ file }: ${ e }` );

			} /* else {

				console.yellow( `Render timeout exceeded in file ${ file }` );

			} */ // TODO: fix this

		}

		const pageElapsed = ( performance.now() - pageStart ) / 1000;

		const screenshot = ( await Image.read( await page.screenshot() ) ).scale( 1 / viewScale );

		if ( page.error !== undefined ) throw new Error( page.error );

		if ( isMakeScreenshot ) {

			/* Make screenshots */

			await screenshot.write( `examples/screenshots/${ file }.jpg`, jpgQuality );

			console.green( `Screenshot generated for file ${ file }` );

		} else {

			/* Diff screenshots */

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

			/* Print results */

			const differentPixels = numDifferentPixels / ( actual.width * actual.height ) * 100;

			if ( differentPixels < maxDifferentPixels ) {

				console.green( `Diff ${ differentPixels.toFixed( 1 ) }% in file: ${ file } (${ pageElapsed.toFixed( 1 ) }s)` );

			} else {

				await screenshot.write( `${ outputDir }/${ file }-actual.jpg`, jpgQuality );
				await expected.write( `${ outputDir }/${ file }-expected.jpg`, jpgQuality );
				await diff.write( `${ outputDir }/${ file }-diff.jpg`, jpgQuality );
				throw new Error( `Diff wrong in ${ differentPixels.toFixed( 1 ) }% of pixels in file: ${ file } (${ pageElapsed.toFixed( 1 ) }s)` );

			}

		}

	} catch ( e ) {

		if ( String( e ).includes( 'WebGPU Device Lost' ) ) {

			console.yellow( `${ e }` );
			console.yellow( 'Restarting browser...' );
			await ctx.restart();

		} else {

			console.red( e );
			failedScreenshots.push( file );

		}

	} finally {

		page.file = undefined; // release lock

	}

}

function close( exitCode = 1 ) {

	console.log( 'Closing...' );

	if ( browserServer ) browserServer.close();
	server.close();
	process.exit( exitCode );

}

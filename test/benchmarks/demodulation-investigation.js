import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import puppeteer from 'puppeteer';
// Run from the repository root after npm run build. See demodulation-investigation.md.
const root = process.cwd();
let original = await readFile( process.env.BENCH_SOURCE || path.join( root, 'examples/webgpu_postprocessing_denoise_demodulation.html' ), 'utf8' );
// Experiment: emit demodulated lighting directly from the scene MRT, avoiding RTT work.
if ( process.env.BENCH_VARIANT?.includes( 'mrt' ) ) {

	original = original.replace( 'albedo: vec4( diffuseColor.rgb, 1 ),', 'lighting: vec4( vec3( noisyLighting ), 1 ),\n\t\t\t\talbedo: vec4( diffuseColor.rgb, 1 ),' );
	original = original.replace( /const demodulated = .*;/, 'const demodulated = scenePass.getTextureNode( \'lighting\' );' );

}

if ( process.env.BENCH_VARIANT?.includes( 'no-msaa' ) ) original = original.replace( 'antialias: true', 'antialias: false' );

const server = http.createServer( async ( req, res ) => {

	try {

		const pathname = new URL( req.url, 'http://localhost' ).pathname;
		const file = path.resolve( root, '.' + pathname );
		if ( ! file.startsWith( root + path.sep ) ) throw new Error( 'Outside repository' );
		const types = { '.js': 'text/javascript', '.html': 'text/html', '.css': 'text/css', '.json': 'application/json' };
		res.setHeader( 'Content-Type', types[ path.extname( file ) ] || 'application/octet-stream' );
		if ( pathname.endsWith( 'webgpu_postprocessing_denoise_demodulation.html' ) ) {

			let html = original.replace( 'renderer.inspector = new Inspector();', 'if (!new URLSearchParams(location.search).has("noInspector")) renderer.inspector = new Inspector();' );
			html = html.replace( 'const gui = renderer.inspector.createParameters( \'Demodulation settings\' );', 'const gui = renderer.inspector ? renderer.inspector.createParameters( \'Demodulation settings\' ) : { add() { return { onChange() { return this; }, name() { return this; } }; } };' );
			html = html.replace( 'renderer.setAnimationLoop( () => {', 'window.bench = { renderer, pipeline, frame, without, filteredLighting, scenePass, outputs: { split, without, withDemodulation, noisy, surfaceColor, filteredLighting } };\n if (false) renderer.setAnimationLoop( () => {' );
			res.end( html );

		} else res.end( await readFile( file ) );

	} catch {

		res.writeHead( 404 ); res.end();

	}

} );
await new Promise( resolve => server.listen( 8097, '127.0.0.1', resolve ) );
let browser;
try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || ( process.platform === 'win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : puppeteer.executablePath() ), headless: false, args: [ '--disable-frame-rate-limit', '--disable-gpu-vsync', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--no-first-run' ] } );
	const page = await browser.newPage();
	await page.setViewport( { width: 1280, height: 720, deviceScaleFactor: 1 } );
	const errors = []; page.on( 'pageerror', error => {

		errors.push( error.message ); console.error( error.message );

	} );
	await page.goto( 'http://127.0.0.1:8097/examples/webgpu_postprocessing_denoise_demodulation.html' );
	await page.waitForFunction( 'window.bench !== undefined' );
	const hardware = await page.evaluate( async () => {

		const adapter = await navigator.gpu.requestAdapter();
		const info = adapter.info;
		const gl = document.createElement( 'canvas' ).getContext( 'webgl2' );
		const ext = gl.getExtension( 'WEBGL_debug_renderer_info' );
		await window.bench.renderer.init(); window.bench.renderer._animation.stop();
		const actualInfo = window.bench.renderer.backend.device.adapterInfo;
		return { actualAdapter: actualInfo ? { vendor: actualInfo.vendor, architecture: actualInfo.architecture, device: actualInfo.device, description: actualInfo.description, isFallbackAdapter: actualInfo.isFallbackAdapter } : null, adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter }, webgl: gl.getParameter( ext.UNMASKED_RENDERER_WEBGL ), backend: window.bench.renderer.backend.constructor.name, userAgent: navigator.userAgent };

	} );
	console.log( JSON.stringify( hardware ) );
	if ( /swiftshader|llvmpipe|lavapipe|software/i.test( JSON.stringify( hardware ) ) || hardware.adapter.isFallbackAdapter || hardware.actualAdapter?.isFallbackAdapter ) throw Error( 'Hardware GPU required' );
	if ( process.env.BENCH_RADIUS ) await page.evaluate( radius => {

		window.bench.without.radius.value = radius; window.bench.filteredLighting.radius.value = radius;

	}, Number( process.env.BENCH_RADIUS ) );
	const results = [];
	for ( const [ width, height ] of JSON.parse( process.env.BENCH_SIZES || '[[1280,720],[1920,1080],[3840,2160]]' ) ) {

		await page.setViewport( { width, height, deviceScaleFactor: 1 } );
		for ( let repeat = 0; repeat < Number( process.env.BENCH_REPEATS || 2 ); repeat ++ ) for ( const view of ( process.env.BENCH_VIEWS || 'noisy,without,withDemodulation,split' ).split( ',' ) ) {

			const result = await page.evaluate( async ( { width, height, view, repeat } ) => {

				const { renderer, pipeline, outputs, frame } = window.bench;
				renderer.setSize( width, height );
				pipeline.outputNode = outputs[ view ]; pipeline.needsUpdate = true;
				const queue = renderer.backend.device.queue;
				await queue.onSubmittedWorkDone();
				const run = async milliseconds => {

					const start = performance.now(); let completed = 0;
					do {

						for ( let i = 0; i < 8; i ++ ) {

							renderer.info.reset(); renderer._nodes.nodeFrame.update(); renderer.info.frame = renderer._nodes.nodeFrame.frameId; frame.value ++; pipeline.render();

						}

						await queue.onSubmittedWorkDone(); completed += 8;

					} while ( performance.now() - start < milliseconds );

					return { completed, elapsedMs: performance.now() - start };

				};

				await run( 1500 );
				const measurement = await run( 5000 );
				return { width, height, view, repeat, memoryBytes: renderer.info.memory.total, renderCalls: renderer.info.render.frameCalls, drawCalls: renderer.info.render.drawCalls, ...measurement, fps: measurement.completed * 1000 / measurement.elapsedMs };

			}, { width, height, view, repeat } );
			results.push( result ); console.log( JSON.stringify( result ) );

		}

	}

	if ( errors.length ) throw new Error( errors.join( '\n' ) );
	if ( process.env.BENCH_SHOT ) await page.screenshot( { path: process.env.BENCH_SHOT } );
	await writeFile( process.env.BENCH_OUTPUT || 'demodulation-benchmark.json', JSON.stringify( { hardware, errors, variant: process.env.BENCH_VARIANT || 'example', source: process.env.BENCH_SOURCE || 'working tree', radius: Number( process.env.BENCH_RADIUS || 1.5 ), method: { batchSize: 8, warmupMs: 1500, measureMs: 5000 }, results }, null, 2 ) );

} finally {

	if ( browser ) await browser.close(); server.close();

}

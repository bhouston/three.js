// Labelled 1080p comparison animations for the Hi-Z / adaptive SSR PR.
// Usage: PUPPETEER_EXECUTABLE_PATH=/path/to/Chrome node capture.mjs <three.js checkout> <dir with benchmark JSON> <out dir>
// Needs hardware WebGPU and img2webp (libwebp). Labels take their times from
// benchmark-64.json and benchmark-96-occlusion.json; adaptive times include Hi-Z creation.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { readFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';

const [ root, data, out ] = process.argv.slice( 2 ).map( path => path.replace( /\/$/, '' ) );
const puppeteer = createRequire( `${ root }/package.json` )( 'puppeteer' );
const { createServer } = await import( pathToFileURL( `${ root }/utils/server.js` ) );
const load = async name => Object.fromEntries( JSON.parse( await readFile( `${ data }/${ name }`, 'utf8' ) ).map( entry => [ entry.example, entry ] ) );
const b64 = await load( 'benchmark-64.json' ), b96 = await load( 'benchmark-96-occlusion.json' );
const ms = ( entry, name, hiZ ) => ( entry.modes.find( mode => mode.name === name ).impliedMs + ( hiZ ? entry.creation.impliedMs : 0 ) ).toFixed( 1 );

const server = createServer( { root: `${ root }/` } );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist' ] } );
await mkdir( out, { recursive: true } );
try {

	for ( const example of [ 'mirror', 'stochastic', 'animation' ] ) {

		const e64 = b64[ example ], e96 = b96[ example ], defaults = e64.exampleDefaults;
		const [ approximateIterations, approximateSteps ] = e64.approximate.split( '/' ).map( Number );
		const base = { adaptive: false, maxCandidates: 1, occlusionSkipping: false, approximateSteps: 0, adaptiveSteps: 64 };
		const frames = [
			{ title: 'Linear SSR, default quality', detail: `quality ${ defaults.quality }, max distance ${ defaults.maxDistance } · ${ ms( e64, 'Non-adaptive SSR (example defaults)' ) } ms`, set: { ...base, ...defaults } },
			{ title: `Adaptive SSR, approximate, ${ approximateIterations } iterations`, detail: `Hi-Z traversal, ${ approximateSteps } approximate steps, max distance ${ defaults.maxDistance } · ${ ms( e64, 'Adaptive SSR, approximate (example defaults)', true ) } ms`, set: { ...base, ...defaults, adaptive: true, adaptiveSteps: approximateIterations, approximateSteps } },
			{ title: 'Linear SSR, comparable quality', detail: `quality 1, binary refine, max distance 50 · ${ ms( e64, 'Non-adaptive SSR' ) } ms`, set: { ...base, quality: 1, binaryRefine: true, maxDistance: 50 } },
			{ title: 'Adaptive SSR, 64 iterations', detail: `Hi-Z traversal, max distance 50 · ${ ms( e64, 'Adaptive SSR (1 candidate)', true ) } ms`, set: { ...base, ...defaults, maxDistance: 50, adaptive: true } },
			{ title: 'Adaptive SSR + occlusion skipping, 96 iterations', detail: `Hi-Z traversal passing behind occluders, max distance 50 · ${ ms( e96, 'Adaptive SSR + occlusion skipping (1 candidate)', true ) } ms`, set: { ...base, ...defaults, maxDistance: 50, adaptive: true, adaptiveSteps: 96, occlusionSkipping: true } }
		];

		const page = await browser.newPage();
		await page.setViewport( { width: 1920, height: 1080, deviceScaleFactor: 1 } );
		page.on( 'pageerror', error => console.error( error.message ) );
		const filename = `webgpu_postprocessing_ssr_${ example }.html`;
		let html = await readFile( `${ root }/examples/${ filename }`, 'utf8' );
		html = html.replaceAll( '../build/three.webgpu.js', '../src/Three.WebGPU.js' ).replaceAll( '../build/three.tsl.js', '../src/Three.TSL.js' );
		const expose = `window.__ssrCapture = { renderer, all: params, params: ${ example === 'stochastic' ? 'params.ssr' : 'params' }, apply: ${ example === 'stochastic' ? 'updateTracing' : 'updateParameters' } };`;
		html = html.replace( example === 'stochastic' ? 'updateOutputNode();\n' : 'updateOutput();\n', match => `${ match }${ expose }\n` );
		await page.setRequestInterception( true );
		page.on( 'request', request => {

			if ( new URL( request.url() ).pathname === `/examples/${ filename }` ) request.respond( { contentType: 'text/html', body: html } );
			else request.continue();

		} );
		await page.goto( `http://127.0.0.1:${ server.address().port }/examples/${ filename }` );
		await page.waitForFunction( () => window.__ssrCapture !== undefined, { timeout: 120000 } );
		await page.waitForNetworkIdle( { idleTime: 500, timeout: 120000 } );
		const adapter = await page.evaluate( () => {

			const { renderer, all } = window.__ssrCapture;
			if ( renderer.backend.isWebGPUBackend !== true ) throw new Error( 'Hardware WebGPU is required.' );
			all.animate = false; // the animation example: hold one pose for every frame
			renderer.inspector.domElement.style.display = 'none';
			document.getElementById( 'info' ).style.display = 'none';
			const label = document.createElement( 'div' );
			label.id = 'label';
			label.style.cssText = 'position:fixed;left:40px;top:36px;padding:16px 26px 18px;border-radius:8px;background:rgba(16,18,26,0.9);color:#fff;font-family:Helvetica,Arial,sans-serif;z-index:100000;text-align:left';
			label.innerHTML = '<div id="label-title" style="font-size:34px;font-weight:700;line-height:44px"></div><div id="label-detail" style="font-size:20.5px;line-height:30px;color:#e5e7eb"></div>';
			document.body.appendChild( label );
			const info = renderer.backend.device.adapterInfo;
			return `${ info.vendor } ${ info.architecture } fallback=${ info.isFallbackAdapter }`;

		} );
		console.log( example, adapter );
		if ( /fallback=true|swiftshader|llvmpipe|lavapipe/i.test( adapter ) ) throw new Error( 'Software adapters are not allowed.' );

		const files = [];
		for ( const [ index, frame ] of frames.entries() ) {

			await page.evaluate( ( { title, detail, set } ) => {

				const { params, apply } = window.__ssrCapture;
				Object.assign( params, set );
				apply();
				document.getElementById( 'label-title' ).textContent = title;
				document.getElementById( 'label-detail' ).textContent = detail;

			}, frame );
			await new Promise( resolve => setTimeout( resolve, 4000 ) ); // recompile, then let temporal filters settle
			files.push( `${ out }/${ example }-${ index }.png` );
			await page.screenshot( { path: files.at( - 1 ) } );

		}

		await page.close();
		execFileSync( 'img2webp', [ '-loop', '0', '-lossy', '-q', '82', '-d', '2000', ...files, '-o', `${ out }/${ example }.webp` ] );

	}

} finally {

	await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

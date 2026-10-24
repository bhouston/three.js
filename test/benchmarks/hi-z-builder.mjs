/* global navigator, performance */
// Compare a supplied previous builder with the working tree on a physical GPU.
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';
import { readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import console from 'node:console';
import assert from 'node:assert/strict';
const baseline = await readFile( process.argv[ 2 ], 'utf8' );
const importMap = ( await readFile( 'test/webgpu/ssr-hi-z.html', 'utf8' ) ).match( /<script type="importmap">[\s\S]*?<\/script>/ )[ 0 ];
const output = process.argv[ 3 ] || '/tmp/three-hi-z-builder.json';
const server = createServer();
let browser;
try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless: false,
		args: [ '--disable-software-rasterizer', '--disable-gpu-vsync', '--disable-frame-rate-limit' ]
	} );
	const page = await browser.newPage();
	page.on( 'console', message => { if ( message.type() === 'log' ) console.log( message.text() ); } );
	await page.setRequestInterception( true );
	page.on( 'request', request => request.url().endsWith( '/HiZNode.js?baseline' )
		? request.respond( { status: 200, contentType: 'application/javascript', body: baseline } )
		: request.url().endsWith( '/test/webgpu/ssr-hi-z.html' )
			? request.respond( { status: 200, contentType: 'text/html', body: `<!DOCTYPE html><html><head>${ importMap }</head><body></body></html>` } ) : request.continue() );
	await page.goto( `http://localhost:${ server.address().port }/test/webgpu/ssr-hi-z.html` );
	const report = await page.evaluate( async () => {

		const THREE = await import( 'three/webgpu' );
		const { texture } = await import( 'three/tsl' );
		const before = await import( '/examples/jsm/tsl/display/HiZNode.js?baseline' );
		const after = await import( '/examples/jsm/tsl/display/HiZNode.js' );
		const adapter = await navigator.gpu.requestAdapter();
		const info = adapter?.info;
		const hardware = info && [ info.vendor, info.architecture, info.device, info.description ].join( ' ' );
		if ( ! hardware?.trim() || adapter.isFallbackAdapter || info.isFallbackAdapter || /software|swiftshader|llvmpipe|lavapipe/i.test( hardware ) ) throw new Error( 'Hardware adapter required' );
		const device = await adapter.requestDevice();
		const renderer = new THREE.WebGPURenderer( { device } );
		await renderer.init();
		const errors = [];
		device.addEventListener( 'uncapturederror', event => errors.push( event.error.message ) );
		const results = [];
		try {

			for ( const [ width, height ] of [ [ 3840, 2160 ], [ 1920, 1080 ], [ 801, 603 ] ] ) {

				const data = Float32Array.from( { length: width * height }, ( _, i ) => ( ( Math.imul( i, 1664525 ) + 1013904223 ) >>> 8 ) / 0x1000000 );
				const depth = new THREE.DataTexture( data, width, height, THREE.RedFormat, THREE.FloatType );
				depth.needsUpdate = true;
				const builders = { before: before.hiZ( texture( depth ) ), after: after.hiZ( texture( depth ) ) };
				try {

					const images = {};
					for ( const [ name, pyramid ] of Object.entries( builders ) ) {

						pyramid.updateBefore( { renderer } );
						await device.queue.onSubmittedWorkDone();
						images[ name ] = await renderer.backend.copyTextureToBuffer( pyramid.texture, 0, 0, pyramid.texture.image.width, pyramid.texture.image.height, 0 );

					}
					for ( const level of builders.before.levels ) for ( let y = 0; y < level.height; y ++ ) for ( let x = 0; x < level.width; x ++ ) {

						const index = ( ( level.offset + y ) * width + x ) * 2;
						if ( images.before[ index ] !== images.after[ index ] || images.before[ index + 1 ] !== images.after[ index + 1 ] ) throw new Error( 'Builder bounds differ' );

					}
					for ( const name of [ 'before', 'after', 'after', 'before' ] ) {

						const pyramid = builders[ name ];
						const run = async duration => {

							let completed = 0;
							const start = performance.now();
							do {

								for ( let i = 0; i < 4; i ++ ) pyramid.updateBefore( { renderer } );
								await device.queue.onSubmittedWorkDone();
								completed += 4;

							} while ( performance.now() - start < duration );
							const seconds = ( performance.now() - start ) / 1000;
							return { completed, seconds, buildsPerSecond: completed / seconds };

						};
						await device.queue.onSubmittedWorkDone();
						await run( 1000 );
						await device.queue.onSubmittedWorkDone();
						const result = await run( 5000 );
						await device.queue.onSubmittedWorkDone();
						results.push( { width, height, name, dispatches: pyramid._computeNode.length, intermediateBytes: pyramid._levelTextures.reduce( ( sum, t ) => sum + t.image.width * t.image.height * 8, 0 ), ...result } );

						console.log( JSON.stringify( results.at( - 1 ) ) );
					}

				} finally {

					for ( const builder of Object.values( builders ) ) builder.dispose();
					depth.dispose();

				}

			}

		} finally {

			renderer.dispose();

		}
		return { hardware, results, errors };

	} );
	await writeFile( output, JSON.stringify( { browser: await browser.version(), ...report }, null, 2 ) );
	console.log( report );
	assert.deepEqual( report.errors, [] );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

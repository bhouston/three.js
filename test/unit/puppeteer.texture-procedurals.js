// Run procedural addon tests and capture the demo using a verified physical WebGL GPU.
// Usage: node test/unit/puppeteer.texture-procedurals.js [artifact-directory]
import puppeteer from 'puppeteer';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createServer } from '../../utils/server.js';

const server = createServer();
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
const origin = `http://127.0.0.1:${ server.address().port }`;
let browser;

try {

	// No GPU override flags or software-renderer environment are supplied.
	browser = await puppeteer.launch( { headless: true } );
	const page = await browser.newPage();
	await page.setViewport( { width: 1280, height: 900, deviceScaleFactor: 1 } );
	page.on( 'pageerror', error => console.error( error.message ) );
	await page.evaluateOnNewDocument( () => {

		window._proceduralFailures = [];
		const timer = setInterval( () => {

			if ( ! window.QUnit ) return;
			clearInterval( timer );
			window.QUnit.on( 'testEnd', test => {

				if ( test.status === 'failed' ) window._proceduralFailures.push( { name: test.fullName.join( ' > ' ), errors: test.errors.map( error => error.message ) } );

			} );

		}, 1 );

	} );
	await page.goto( `${ origin }/test/unit/UnitTestsAddons.html?filter=GLTFTextureProcedurals` );
	await page.waitForFunction( () => window._QUnitStats !== undefined, { timeout: 120000 } );
	const tests = await page.evaluate( () => ( { stats: window._QUnitStats, failures: window._proceduralFailures } ) );
	console.log( JSON.stringify( tests, null, 2 ) );
	if ( tests.stats.failed > 0 ) process.exitCode = 1;

	if ( process.argv[ 2 ] ) {

		const directory = path.resolve( process.argv[ 2 ] );
		await mkdir( directory, { recursive: true } );
		await writeFile( path.join( directory, 'texture-procedurals-tests.json' ), JSON.stringify( tests, null, 2 ) + '\n' );
		// Source imports avoid generating root build/ outputs during verification.
		await page.setRequestInterception( true );
		page.on( 'request', request => {

			const name = new URL( request.url() ).pathname;
			const sources = { '/build/three.webgpu.js': '../src/Three.WebGPU.js', '/build/three.tsl.js': '../src/Three.TSL.js' };
			if ( sources[ name ] ) {

				request.respond( { status: 200, contentType: 'application/javascript', body: `export * from '${ sources[ name ] }';` } );

			} else {

				request.continue();

			}

		} );
		await page.goto( `${ origin }/examples/webgpu_loader_gltf_texture_procedurals.html` );
		await page.waitForFunction( () => window.textureProceduralsDemo?.ready, { timeout: 60000 } );
		const renderer = await page.evaluate( () => {

			const gl = window.textureProceduralsDemo.renderer.backend.gl;
			const extension = gl.getExtension( 'WEBGL_debug_renderer_info' );
			return extension ? gl.getParameter( extension.UNMASKED_RENDERER_WEBGL ) : null;

		} );
		if ( ! renderer || /swiftshader|llvmpipe|lavapipe|software|microsoft basic render/i.test( renderer ) ) throw new Error( `Physical GPU verification blocked: ${ renderer }` );
		console.log( `Physical GPU renderer: ${ renderer }` );
		await writeFile( path.join( directory, 'texture-procedurals-renderer.json' ), JSON.stringify( { renderer, backend: 'WebGL2', softwareFallback: false }, null, 2 ) + '\n' );
		await page.screenshot( { path: path.join( directory, 'texture-procedurals.png' ) } );
		const exported = await page.evaluate( async () => {

			const { GLTFExporter } = await import( '/examples/jsm/exporters/GLTFExporter.js' );
			const { GLTFTextureProceduralsExporterExtension } = await import( '/examples/jsm/exporters/GLTFTextureProceduralsExporterExtension.js' );
			const exporter = new GLTFExporter();
			exporter.register( writer => new GLTFTextureProceduralsExporterExtension( writer ) );
			const json = await exporter.parseAsync( window.textureProceduralsDemo.scene );
			const glb = await exporter.parseAsync( window.textureProceduralsDemo.scene, { binary: true } );
			return { json, glb: Array.from( new Uint8Array( glb ) ) };

		} );
		await writeFile( path.join( directory, 'texture-procedurals-export.gltf' ), JSON.stringify( exported.json, null, 2 ) + '\n' );
		await writeFile( path.join( directory, 'texture-procedurals-export.glb' ), Buffer.from( exported.glb ) );
		const colorPixels = await page.evaluate( async () => {

			const THREE = await import( 'three/webgpu' );
			const { createTextureProceduralMaterial } = await import( '/examples/jsm/utils/GLTFTextureProcedurals.js' );
			const image = new THREE.DataTexture( new Uint8Array( [ 128, 128, 128, 255 ] ), 1, 1 );
			image.needsUpdate = true;
			const graph = {
				nodetype: 'nodegraph', type: 'multioutput', inputs: {},
				nodes: [ {
					nodetype: 'image', type: 'color3', colorspace: 'srgb_texture',
					inputs: { file: { nodetype: 'input', type: 'filename', value: [ 0 ] } },
					outputs: { out: { nodetype: 'output', type: 'color3' } }
				}, {
					nodetype: 'image', type: 'color3',
					inputs: { file: { nodetype: 'input', type: 'filename', value: [ 0 ] } },
					outputs: { out: { nodetype: 'output', type: 'color3' } }
				} ],
				outputs: { decoded: { nodetype: 'output', type: 'color3', node: 0, output: 'out' }, raw: { nodetype: 'output', type: 'color3', node: 1, output: 'out' } }
			};
			const scene = new THREE.Scene();
			for ( const [ index, output ] of [ 'decoded', 'raw' ].entries() ) {

				const material = await createTextureProceduralMaterial( new THREE.MeshBasicMaterial(), { procedurals: [ graph ] }, { baseColorTexture: { index: 0, output } }, { textures: [ image ] } );
				const mesh = new THREE.Mesh( new THREE.PlaneGeometry( 2, 2 ), material );
				mesh.position.x = index * 2 - 1;
				scene.add( mesh );

			}

			const camera = new THREE.OrthographicCamera( - 2, 2, 1, - 1, 0.1, 10 );
			camera.position.z = 3;
			const renderer = window.textureProceduralsDemo.renderer;
			await renderer.renderAsync( scene, camera );
			const gl = renderer.backend.gl;
			const decoded = new Uint8Array( 4 );
			const raw = new Uint8Array( 4 );
			gl.readPixels( gl.drawingBufferWidth / 4, gl.drawingBufferHeight / 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, decoded );
			gl.readPixels( gl.drawingBufferWidth * 3 / 4, gl.drawingBufferHeight / 2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, raw );
			return { decoded: Array.from( decoded ), raw: Array.from( raw ), expected: { decoded: 128, raw: 188 } };

		} );
		await writeFile( path.join( directory, 'texture-procedurals-color-pixels.json' ), JSON.stringify( colorPixels, null, 2 ) + '\n' );
		if ( colorPixels.decoded.slice( 0, 3 ).some( value => Math.abs( value - 128 ) > 3 ) || colorPixels.raw.slice( 0, 3 ).some( value => Math.abs( value - 188 ) > 3 ) ) throw new Error( `Procedural image color interpretation failed: ${ JSON.stringify( colorPixels ) }` );

	}

} finally {

	if ( browser ) await browser.close();
	await new Promise( resolve => server.close( resolve ) );

}

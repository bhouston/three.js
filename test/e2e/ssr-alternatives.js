// Run after npm run build:
// node test/e2e/ssr-alternatives.js [artifact-directory]
// This is a hardware rendering regression check, not a throughput benchmark.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';

const root = fileURLToPath( new URL( '../../', import.meta.url ) );
const artifacts = path.resolve( process.argv[ 2 ] || process.env.SWARM_ARTIFACTS || path.join( os.tmpdir(), 'three-ssr-alternatives' ) );
const report = { status: 'failed', viewport: [ 320, 240 ], hardware: null, cases: [], errors: [] };
const server = createServer( { root } );
let browser;

try {

	await mkdir( artifacts, { recursive: true } );
	await new Promise( ( resolve, reject ) => {

		server.once( 'error', reject );
		server.listen( 0, '127.0.0.1', resolve );

	} );
	const origin = `http://127.0.0.1:${ server.address().port }`;
	browser = await puppeteer.launch( { headless: true } );
	const page = await browser.newPage();
	await page.setViewport( { width: 320, height: 240, deviceScaleFactor: 1 } );
	await page.setRequestInterception( true );
	page.on( 'request', request => {

		if ( new URL( request.url() ).pathname === '/favicon.ico' ) {

			request.respond( { status: 204 } );

		} else {

			request.continue();

		}

	} );
	page.on( 'pageerror', error => report.errors.push( String( error ) ) );
	page.on( 'console', message => {

		if ( [ 'error', 'warn' ].includes( message.type() ) && ! report.errors.includes( message.text() ) ) report.errors.push( message.text() );

	} );
	await page.goto( origin );
	await page.setContent( '<link rel="icon" href="data:,"><script type="importmap">{"imports":{"three":"/build/three.webgpu.js","three/webgpu":"/build/three.webgpu.js","three/tsl":"/build/three.tsl.js"}}</script>' );
	report.hardware = await page.evaluate( async () => {

		const adapter = await navigator.gpu?.requestAdapter();
		if ( ! adapter ) throw new Error( 'No hardware WebGPU adapter is available.' );
		const info = adapter.info;
		const adapterInfo = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false };
		const canvas = document.createElement( 'canvas' );
		const gl = canvas.getContext( 'webgl2' );
		const extension = gl?.getExtension( 'WEBGL_debug_renderer_info' );
		const renderer = extension ? gl.getParameter( extension.UNMASKED_RENDERER_WEBGL ) : '';
		gl?.getExtension( 'WEBGL_lose_context' )?.loseContext();
		return { adapter: adapterInfo, webglRenderer: renderer };

	} );
	const software = /swiftshader|llvmpipe|lavapipe|software|microsoft basic|\bwarp\b/i;
	const physical = /nvidia|geforce|quadro|radeon|\bamd\b|intel|apple|qualcomm|adreno|mali|samsung|imagination|10de|1002|8086|106b/i;
	assert.equal( report.hardware.adapter.isFallbackAdapter, false, 'WebGPU must use a physical adapter' );
	assert.ok( ! software.test( JSON.stringify( report.hardware ) ), 'Software rendering is not accepted' );
	assert.ok( physical.test( JSON.stringify( report.hardware.adapter ) ), 'WebGPU adapter must identify physical hardware' );
	assert.ok( physical.test( report.hardware.webglRenderer ), 'WebGL renderer must identify physical hardware' );

	for ( const cameraType of [ 'perspective', 'orthographic' ] ) {

		for ( const reversed of [ false, true ] ) {

			const result = await page.evaluate( async ( { cameraType, reversed } ) => {

				const THREE = await import( '/build/three.webgpu.js' );
				const tsl = await import( '/build/three.tsl.js' );
				const { powerLobeSSR } = await import( '/examples/jsm/tsl/display/PowerLobeSSRNode.js' );
				// Supply the explicitly inspected adapter's device to the renderer.
				const adapter = await navigator.gpu.requestAdapter();
				if ( ! adapter || adapter.info.isFallbackAdapter === true || adapter.isFallbackAdapter === true ) throw new Error( 'Hardware WebGPU adapter unavailable.' );
				const info = adapter.info;
				const deviceInfo = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
				const device = await adapter.requestDevice( { requiredFeatures: [ ...adapter.features ] } );
				const gpuErrors = [];
				device.addEventListener( 'uncapturederror', event => gpuErrors.push( event.error.message ) );
				const renderer = new THREE.WebGPURenderer( { reversedDepthBuffer: reversed, device } );
				renderer.setSize( 320, 240 );
				await renderer.init();
				if ( renderer.backend.isWebGPUBackend !== true ) throw new Error( 'Expected hardware WebGPU backend.' );
				const camera = cameraType === 'perspective' ? new THREE.PerspectiveCamera( 50, 4 / 3, 0.1, 30 ) : new THREE.OrthographicCamera( - 3, 3, 2.25, - 2.25, 0.1, 30 );
				camera.position.set( 3, 2, 3 );
				camera.lookAt( 0, - 0.25, 0 );
				camera.updateMatrixWorld();
				const scene = new THREE.Scene();
				const floor = new THREE.Mesh( new THREE.PlaneGeometry( 8, 8 ), new THREE.MeshStandardMaterial( { color: 0xffffff, metalness: 1, roughness: 0 } ) );
				floor.rotation.x = - Math.PI / 2;
				floor.position.y = - 1;
				scene.add( floor );
				const box = new THREE.Mesh( new THREE.BoxGeometry( 1, 1, 1 ), new THREE.MeshStandardMaterial( { color: 0xffffff, emissive: 0xff4400, emissiveIntensity: 20 } ) );
				box.position.y = - 0.25;
				scene.add( box );
				const environment = new THREE.DataTexture( new Float32Array( [ 4, 2, 1, 1, 4, 2, 1, 1 ] ), 2, 1, THREE.RGBAFormat, THREE.FloatType );
				environment.mapping = THREE.EquirectangularReflectionMapping;
				environment.needsUpdate = true;
				const scenePass = tsl.pass( scene, camera );
				scenePass.transparent = false;
				scenePass.setMRT( tsl.mrt( { output: tsl.output, normal: tsl.vec4( tsl.packNormalToRGB( tsl.normalView ), tsl.materialRoughness ) } ) );
				const color = scenePass.getTextureNode( 'output' );
				const depth = scenePass.getTextureNode( 'depth' );
				const packedNormal = scenePass.getTextureNode( 'normal' );
				const normal = tsl.sample( uv => tsl.unpackRGBToNormal( packedNormal.sample( uv ).rgb ) );
				const node = powerLobeSSR( color, depth, normal, { camera, environmentNode: environment, metalnessNode: tsl.float( 1 ), roughnessNode: packedNormal.a } );
				node.maxDistance.value = 5;
				node.steps.value = 32;
				node.thickness.value = 0.08;
				node.seed.value = 7;
				const pipeline = new THREE.RenderPipeline( renderer );
				pipeline.outputNode = node;
				const render = async ( count = 1 ) => {

					for ( let i = 0; i < count; i ++ ) {

						renderer._nodes.nodeFrame.update();
						pipeline.render();

					}

					await device.queue.onSubmittedWorkDone();

				};

				const read = async index => {

					const target = node.getRenderTarget();
					const raw = await renderer.readRenderTargetPixelsAsync( target, 0, 0, target.width, target.height, index );
					return Float32Array.from( raw, THREE.DataUtils.fromHalfFloat );

				};

				const readAll = async () => {

					const outputs = [];
					for ( let index = 0; index < node.getRenderTarget().textures.length; index ++ ) outputs.push( await read( index ) );
					return outputs;

				};

				const nonfiniteCount = outputs => outputs.reduce( ( count, output ) => count + output.reduce( ( sum, value ) => sum + ( Number.isFinite( value ) ? 0 : 1 ), 0 ), 0 );
				try {

					await render( 3 );
					const outputs = await readAll();
					const hit = outputs[ 3 ];
					let hits = 0, badUV = 0, outOfReach = 0;
					for ( let i = 0; i < hit.length; i += 4 ) {

						if ( hit[ i + 3 ] < 0.5 ) continue;
						hits ++;
						if ( hit[ i ] < 0 || hit[ i ] > 1 || hit[ i + 1 ] < 0 || hit[ i + 1 ] > 1 ) badUV ++;
						if ( hit[ i + 2 ] < 0 || hit[ i + 2 ] > 5.01 ) outOfReach ++;

					}

					const result = { cameraType, reversed, deviceInfo, hits, nonfinite: nonfiniteCount( outputs ), badUV, outOfReach, normalization: [], gpuErrors };
					node.maxDistance.value = 0;
					for ( const roughness of [ 0, 0.001, 1 ] ) {

						for ( const rays of [ 1, 2, 4 ] ) {

							floor.material.roughness = roughness;
							box.material.roughness = roughness;
							node.rayCount.value = rays;
							node.resetHistory();
							await render();
							const outputs = await readAll();
							const incident = outputs[ 1 ], metadata = outputs[ 2 ];
							let valid = 0, error = 0, badCoverage = 0;
							for ( let i = 0; i < incident.length; i += 4 ) {

								if ( metadata[ i + 3 ] > 0 ) {

									valid ++;
									error = Math.max( error, Math.abs( incident[ i ] - 4 ), Math.abs( incident[ i + 1 ] - 2 ), Math.abs( incident[ i + 2 ] - 1 ) );

								}

								if ( incident[ i + 3 ] !== 0 || metadata[ i ] !== 0 || outputs[ 3 ][ i + 3 ] !== 0 ) badCoverage ++;

							}

							result.normalization.push( { roughness, rays, valid, error, badCoverage, nonfinite: nonfiniteCount( outputs ) } );

						}

					}

					return result;

				} finally {

					await device.queue.onSubmittedWorkDone();
					node.dispose();
					scenePass.dispose();
					pipeline.dispose();
					renderer.dispose();
					floor.geometry.dispose();
					box.geometry.dispose();
					floor.material.dispose();
					box.material.dispose();
					environment.dispose();
					device.destroy();

				}

			}, { cameraType, reversed } );
			report.cases.push( result );
			assert.ok( physical.test( JSON.stringify( result.deviceInfo ) ) && ! software.test( JSON.stringify( result.deviceInfo ) ), 'Fixture renderer uses the inspected physical adapter' );
			assert.ok( result.hits > 0, `${ cameraType } reversed=${ reversed }: fixture must produce screen hits` );
			assert.equal( result.nonfinite, 0, 'Every output attachment must remain finite' );
			assert.equal( result.badUV, 0, 'Valid hit UVs stay onscreen' );
			assert.equal( result.outOfReach, 0, 'Hit distances respect configured reach' );
			assert.deepEqual( result.gpuErrors, [], 'No WebGPU validation errors' );
			for ( const entry of result.normalization ) {

				assert.ok( entry.valid > 0, 'Constant-HDR fixture must launch valid rays' );
				assert.equal( entry.error, 0, 'Constant HDR remains constant for every roughness and ray count' );
				assert.equal( entry.badCoverage, 0, 'Zero reach always has zero hit coverage/validity, including background' );
				assert.equal( entry.nonfinite, 0, 'All normalization outputs remain finite' );

			}

			console.log( `PASS ${ cameraType }, reversed=${ reversed }: ${ result.hits } hits; nine normalization cases` );

		}

	}

	assert.deepEqual( report.errors, [], 'Browser must report no errors or warnings' );
	report.status = 'passed';

} catch ( error ) {

	report.error = String( error.stack || error );
	console.error( report.error );
	process.exitCode = 1;

} finally {

	await browser?.close();
	if ( server.listening ) await new Promise( ( resolve, reject ) => server.close( error => error ? reject( error ) : resolve() ) );
	await mkdir( artifacts, { recursive: true } );
	const reportPath = path.join( artifacts, 'ssr-alternatives-regression.json' );
	await writeFile( reportPath, JSON.stringify( report, null, 2 ) + '\n' );
	console.log( `SSR regression report: ${ reportPath }` );

}

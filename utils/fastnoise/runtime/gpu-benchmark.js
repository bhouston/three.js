// Sustained, GPU-completed scalar sampling throughput. No Three.js build required.
// node utils/fastnoise/runtime/gpu-benchmark.js --root . --output report.json
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const args = process.argv.slice( 2 );
const option = ( name, fallback ) => args.includes( name ) ? args[ args.indexOf( name ) + 1 ] : fallback;
const root = path.resolve( option( '--root', '.' ) );
const output = path.resolve( option( '--output', 'runtime-gpu.json' ) );
const seconds = Number( option( '--seconds', 5 ) );
if ( ! ( seconds >= 5 ) ) throw new Error( 'Use at least five seconds per method.' );
const require = createRequire( path.join( root, 'package.json' ) );
const puppeteer = require( 'puppeteer' );
const { PNG } = require( 'pngjs' );
const moduleURL = file => pathToFileURL( path.join( root, 'utils/fastnoise/runtime', file ) ).href;
const { proceduralSamplers, wgslSamplers } = await import( moduleURL( 'procedural.js' ) );
const phase = await import( moduleURL( 'phase-tile.js' ) );
const atlas = PNG.sync.read( await fs.readFile( path.join( root, 'examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png' ) ) );
const checks = [[ 0, 0, 0 ], [ 7, 19, 1 ], [ 63, 91, 31 ], [ 129, 253, 32 ], [ 511, 511, 1023 ]];
const profiles = Object.entries( wgslSamplers ).map( ( [ id, wgsl ] ) => ( { id, wgsl, expected: checks.map( xyz => proceduralSamplers[ id ]( ...xyz, 1 ) ), allocationBytes: 0 } ) );
for ( const passes of [ 2, 8 ] ) {

	const tile = phase.generate( { size: 32, seed: 1, passes } );
	profiles.push( { id: `phase-tile-32-${ passes }`, allocationBytes: tile.byteLength, tile: Array.from( tile ), expected: checks.map( xyz => phase.sample( tile, ...xyz ) ), wgsl: 'fn sample(x:u32,y:u32,frame:u32,seed:u32)->f32 { return fract(tile[(y & 31u)*32u+(x & 31u)]+f32((frame*2654435769u)>>8u)/16777216.0); }' } );

}

profiles.push( {
	id: 'fast-texture-reference', allocationBytes: atlas.width * atlas.height * 4, atlas: true,
	wgsl: 'fn sample(x:u32,y:u32,frame:u32,seed:u32)->f32 { let cycle=(frame/32u)%128u; let p=vec2u((x+37u*cycle)%128u, (y+71u*cycle)%128u+128u*(frame%32u)); return (textureLoad(atlas, vec2i(p), 0).r*255.0+0.5)/256.0; }',
	expected: checks.map( ( [ x, y, frame ] ) => {

		const cycle = Math.floor( frame / 32 ) % 128;
		return ( atlas.data[ ( ( ( y + 71 * cycle ) % 128 + 128 * ( frame % 32 ) ) * atlas.width + ( x + 37 * cycle ) % 128 ) * 4 ] + .5 ) / 256;

	} )
} );
const server = http.createServer( ( request, response ) => response.setHeader( 'Content-Type', 'text/html' ).end( '<!doctype html><title>Hardware sampling benchmark</title>' ) );
await new Promise( resolve => server.listen( 0, '127.0.0.1', resolve ) );
let browser;
try {

	browser = await puppeteer.launch( { executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [ '--enable-unsafe-webgpu', '--disable-software-rasterizer', '--disable-frame-rate-limit', '--disable-gpu-vsync' ], protocolTimeout: 600000 } );
	const page = await browser.newPage();
	await page.goto( `http://127.0.0.1:${ server.address().port }` );
	const report = await page.evaluate( async ( { profiles, checks, seconds, pixels, width, height } ) => {

		const adapter = await navigator.gpu?.requestAdapter( { powerPreference: 'high-performance' } );
		const info = adapter?.info;
		if ( ! info || info.isFallbackAdapter || ! /nvidia|amd|intel|apple|qualcomm/i.test( info.vendor ) || /swiftshader|software|llvmpipe|lavapipe/i.test( JSON.stringify( info ) ) ) throw new Error( 'Physical WebGPU adapter unavailable.' );
		const device = await adapter.requestDevice();
		const errors = [];
		device.addEventListener( 'uncapturederror', event => errors.push( event.error.message ) );
		const count = 512 * 512, batch = 8;
		const results = [];
		for ( const profile of profiles ) {

			const output = device.createBuffer( { size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC } );
			const readback = device.createBuffer( { size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
			let tile, texture;
			let declaration = '';
			if ( profile.tile ) {

				tile = device.createBuffer( { size: profile.tile.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST } );
				device.queue.writeBuffer( tile, 0, new Float32Array( profile.tile ) );
				declaration = '@group(0) @binding(1) var<storage,read> tile: array<f32>;';

			} else if ( profile.atlas ) {

				texture = device.createTexture( { size: [ width, height ], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
				device.queue.writeTexture( { texture }, new Uint8Array( pixels ), { bytesPerRow: width * 4 }, [ width, height ] );
				declaration = '@group(0) @binding(1) var atlas: texture_2d<f32>;';

			}

			const code = `@group(0) @binding(2) var<uniform> params:vec4u; @group(0) @binding(0) var<storage,read_write> output:array<f32>; ${ declaration } ${ profile.wgsl }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3u,@builtin(workgroup_id) group:vec3u) { let i=id.x; if(i<262144u) { output[i]=sample(i%512u,i/512u,params.x,1u); } }
@compute @workgroup_size(1) fn check() { ${ checks.map( ( [ x, y, frame ], i ) => `output[${ i }]=sample(${ x }u,${ y }u,${ frame }u,1u);` ).join( '\n' ) } }`;
			const module = device.createShaderModule( { code } );
			const compilation = await module.getCompilationInfo();
			if ( compilation.messages.some( message => message.type === 'error' ) ) throw new Error( profile.id + ': ' + compilation.messages.map( message => message.message ).join( ';' ) );
			const pipeline = await device.createComputePipelineAsync( { layout: 'auto', compute: { module, entryPoint: 'main' } } );
			const checkPipeline = await device.createComputePipelineAsync( { layout: 'auto', compute: { module, entryPoint: 'check' } } );
			const entries = [ { binding: 0, resource: { buffer: output } } ];
			if ( tile ) entries.push( { binding: 1, resource: { buffer: tile } } );
			if ( texture ) entries.push( { binding: 1, resource: texture.createView() } );
			const frameBuffers = Array.from( { length: batch }, ( _, frame ) => {

				const buffer = device.createBuffer( { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } ); device.queue.writeBuffer( buffer, 0, new Uint32Array( [ frame, 0, 0, 0 ] ) ); return buffer;

			} );
			const binds = frameBuffers.map( buffer => device.createBindGroup( { layout: pipeline.getBindGroupLayout( 0 ), entries: [ ...entries, { binding: 2, resource: { buffer } } ] } ) );
			const checkBind = device.createBindGroup( { layout: checkPipeline.getBindGroupLayout( 0 ), entries } );
			let encoder = device.createCommandEncoder();
			let pass = encoder.beginComputePass();
			pass.setPipeline( checkPipeline ); pass.setBindGroup( 0, checkBind ); pass.dispatchWorkgroups( 1 ); pass.end();
			encoder.copyBufferToBuffer( output, 0, readback, 0, count * 4 );
			device.queue.submit( [ encoder.finish() ] );
			await readback.mapAsync( GPUMapMode.READ );
			const actual = Array.from( new Float32Array( readback.getMappedRange() ).slice( 0, checks.length ) );
			readback.unmap();
			const maxCPUError = Math.max( ...actual.map( ( value, i ) => Math.abs( value - profile.expected[ i ] ) ) );
			if ( maxCPUError > .002 ) throw new Error( `${ profile.id }: CPU/GPU sample mismatch ${ maxCPUError }` );
			const runBatch = async () => {

				encoder = device.createCommandEncoder();
				for ( let frame = 0; frame < batch; frame ++ ) {

					pass = encoder.beginComputePass(); pass.setPipeline( pipeline ); pass.setBindGroup( 0, binds[ frame ] ); pass.dispatchWorkgroups( 1024 ); pass.end();

				}

				device.queue.submit( [ encoder.finish() ] );
				await device.queue.onSubmittedWorkDone();

			};

			// Two seconds of warmup; each counted batch is drained before counting.
			let start = performance.now();
			while ( performance.now() - start < 2000 ) await runBatch();
			await device.queue.onSubmittedWorkDone();
			let completedFrames = 0;
			start = performance.now();
			while ( performance.now() - start < seconds * 1000 ) {

				await runBatch(); completedFrames += batch;

			}

			const elapsedSeconds = ( performance.now() - start ) / 1000;
			await device.queue.onSubmittedWorkDone();
			results.push( { id: profile.id, allocationBytes: profile.allocationBytes, completedFrames, elapsedSeconds, samplesPerSecond: completedFrames * count / elapsedSeconds, maxCPUError, actual } );
			output.destroy(); readback.destroy(); tile?.destroy(); texture?.destroy(); for ( const buffer of frameBuffers ) buffer.destroy();
			if ( errors.length ) throw new Error( errors.join( '\n' ) );

		}

		device.destroy();
		return { gpu: { vendor: info.vendor, architecture: info.architecture, description: info.description, isFallbackAdapter: info.isFallbackAdapter }, userAgent: navigator.userAgent, workload: '512x512 scalar samples, eight sequential compute dispatches per completion wait; two-second warmup; no vsync or frame cap; end-to-end throughput including submission/completion overhead', results };

	}, { profiles, checks, seconds, pixels: Array.from( atlas.data ), width: atlas.width, height: atlas.height } );
	await fs.mkdir( path.dirname( output ), { recursive: true } );
	await fs.writeFile( output, JSON.stringify( report, null, 2 ) + '\n' );
	console.log( JSON.stringify( report, null, 2 ) );

} finally {

	await browser?.close();
	await new Promise( resolve => server.close( resolve ) );

}

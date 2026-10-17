// Opt-in hardware smoke/lifecycle test: node test/e2e/surface-layout.js <artifact-dir> [hosted-url]
// No throughput measurements. Software adapters are rejected before screenshots count.
import puppeteer from 'puppeteer';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createServer } from '../../utils/server.js';

const artifacts = path.resolve( process.argv[ 2 ] || 'test/e2e/output-screenshots/surface-layout' );
await mkdir( artifacts, { recursive: true } );
const server = createServer();
await new Promise( resolve => server.listen( 8121, resolve ) );
const url = process.argv[ 3 ] || 'http://localhost:8121/examples/webgpu_surface_layout.html';
const result = { url, checks: [], errors: [], warnings: [] };
let browser;

try {

	browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu' ], defaultViewport: { width: 1100, height: 760 } } );
	result.browser = await browser.version();
	const cdp = await browser.target().createCDPSession();
	result.gpu = ( await cdp.send( 'SystemInfo.getInfo' ) ).gpu;
	const page = await browser.newPage();
	page.on( 'pageerror', error => result.errors.push( error.stack ) );
	page.on( 'console', message => {

		if ( message.type() === 'error' ) result.errors.push( message.text() );
		if ( message.type() === 'warn' ) result.warnings.push( message.text() );

	} );
	await page.goto( url, { waitUntil: 'networkidle0' } );
	await page.waitForFunction( () => window.surfaceDemo, { timeout: 60000 } );
	result.adapter = await page.evaluate( () => {

		const { renderer } = window.surfaceDemo;
		const info = renderer.backend.device?.adapterInfo;
		renderer.backend.device.addEventListener( 'uncapturederror', e => console.error( e.error.message ) );
		return { backend: renderer.backend.constructor.name, maxColorAttachments: renderer.backend.device.limits.maxColorAttachments, maxColorAttachmentBytesPerSample: renderer.backend.device.limits.maxColorAttachmentBytesPerSample, ...Object.fromEntries( [ 'vendor', 'architecture', 'device', 'description', 'isFallbackAdapter' ].map( key => [ key, info?.[ key ] ] ) ) };

	} );
	assert.equal( result.adapter.backend, 'WebGPUBackend' );
	assert.equal( result.adapter.isFallbackAdapter, false );
	assert.match( result.adapter.vendor, /nvidia|amd|intel|apple|qualcomm/i );
	assert.doesNotMatch( JSON.stringify( result.adapter ), /swiftshader|llvmpipe|lavapipe|software|basic render/i );
	assert.ok( result.adapter.maxColorAttachments >= 4 );
	assert.ok( result.adapter.maxColorAttachmentBytesPerSample >= 24 );
	console.log( 'Hardware:', result.adapter, result.gpu.devices[ 0 ].deviceString );

	const settle = () => page.evaluate( () => new Promise( resolve => {

		let remaining = 12;
		function next() {

			if ( -- remaining > 0 ) requestAnimationFrame( next );
			else window.surfaceDemo.renderer.backend.device.queue.onSubmittedWorkDone().then( resolve );

		}

		requestAnimationFrame( next );

	} ) );
	const configure = async config => {

		await page.evaluate( values => {

			Object.assign( window.surfaceDemo.params, values );
			window.surfaceDemo.rebuild();
			window.normalReadback?.dispose();
			window.normalReadback = null;

		}, config );
		await settle();

	};

	const inventory = () => page.evaluate( () => {

		const { renderer, state } = window.surfaceDemo;
		const textures = [ ...state.source.renderTarget.textures, state.source.renderTarget.depthTexture ].map( texture => ( {
			name: texture.name, ...renderer.backend.get( texture ).textureDescriptorGPU
		} ) );
		const targets = state.owned.flatMap( node => Object.entries( node ).filter( ( [ , value ] ) => value?.isRenderTarget ).map( ( [ key, target ] ) => {

			const depth = renderer._textures.get( target ).depthTexture;
			return { owner: node.name || node.constructor.name, key, width: target.width, height: target.height, depthBuffer: target.depthBuffer,
				textures: [ ...target.textures, ...( depth ? [ depth ] : [] ) ].map( texture => ( { name: texture.name, ...renderer.backend.get( texture ).textureDescriptorGPU } ) ) };

		} ) );
		return { params: { ...window.surfaceDemo.params }, generation: state.generation, attachments: state.layout.attachmentNames, textures, targets, frameDrawCalls: renderer.info.render.frameCalls, memory: { ...renderer.info.memory } };

	} );
	await settle();
	const rich = await inventory();
	result.checks.push( { name: 'descriptor/rgb8/rich', ...rich } );
	await page.screenshot( { path: path.join( artifacts, 'surface-rich.png' ) } );
	await configure( { layout: 'plain' } );
	const plain = await inventory();
	assert.deepEqual( plain.textures, rich.textures, 'plain and descriptor have identical GPU attachments' );
	assert.equal( plain.memory.renderTargets, rich.memory.renderTargets );
	assert.equal( plain.memory.texturesSize, rich.memory.texturesSize );
	result.checks.push( { name: 'plain/rgb8/rich', ...plain } );
	await configure( { layout: 'descriptor', packing: 'oct16' } );
	result.checks.push( { name: 'descriptor/oct16/rich', ...await inventory() } );

	// Opening Inspector previews must not execute a shared producer twice per frame.
	await page.evaluate( () => {

		window.surfaceCalls = {};
		const source = window.surfaceDemo.state.source;
		const update = source.updateBefore;
		source.updateBefore = function ( frame ) {

			window.surfaceCalls[ frame.frameId ] = ( window.surfaceCalls[ frame.frameId ] || 0 ) + 1;
			return update.call( this, frame );

		};

		const inspector = window.surfaceDemo.renderer.inspector;
		if ( ! inspector.profiler.panel.classList.contains( 'visible' ) ) inspector.profiler.togglePanel();
		inspector.setActiveTab( inspector.viewer );

	} );
	await settle();
	result.inspector = await page.evaluate( () => ( {
		buffers: Array.from( window.surfaceDemo.renderer.inspector.viewer.select.options, o => o.textContent ),
		producerCalls: Object.values( window.surfaceCalls )
	} ) );
	assert.ok( result.inspector.buffers.includes( 'normal Rough' ) );
	assert.ok( result.inspector.buffers.includes( 'Depth' ) );
	assert.ok( result.inspector.producerCalls.length >= 5 );
	assert.ok( result.inspector.producerCalls.every( count => count === 1 ) );
	await page.screenshot( { path: path.join( artifacts, 'surface-inspector.png' ) } );

	// Compare the actual GPU shader-decoded normals, excluding background pixels.
	const normalSamples = [];
	for ( const packing of [ 'rgb8', 'oct16' ] ) {

		await configure( { packing, ao: false, gi: false, aa: false, view: 'normal' } );
		await page.evaluate( async () => {

			const { rtt, vec4 } = await import( 'three/tsl' );
			const demo = window.surfaceDemo;
			const source = demo.state.source;
			window.normalReadback = rtt( vec4( demo.state.layout.nodes.normal, source.getTextureNode( 'depth' ).lessThan( 1 ) ) );
			demo.pipeline.outputNode = window.normalReadback;
			demo.pipeline.needsUpdate = true;

		} );
		await settle();
		normalSamples.push( await page.evaluate( async () => {

			const { DataUtils } = await import( 'three/webgpu' );
			const buffer = await window.surfaceDemo.renderer.readRenderTargetPixelsAsync( window.normalReadback.renderTarget, 500, 340, 64, 64 );
			const values = Array.from( buffer, value => DataUtils.fromHalfFloat( value ) );
			return values;

		} ) );

	}

	let maxAngle = 0, pixels = 0;
	for ( let i = 0; i < normalSamples[ 0 ].length; i += 4 ) {

		const a = normalSamples[ 0 ].slice( i, i + 3 ), b = normalSamples[ 1 ].slice( i, i + 3 );
		if ( normalSamples[ 0 ][ i + 3 ] < 0.5 || normalSamples[ 1 ][ i + 3 ] < 0.5 ) continue;
		const cosine = a.reduce( ( sum, value, axis ) => sum + value * b[ axis ], 0 ) / ( Math.hypot( ...a ) * Math.hypot( ...b ) );
		maxAngle = Math.max( maxAngle, Math.acos( Math.min( 1, Math.max( - 1, cosine ) ) ) );
		pixels ++;

	}

	assert.ok( pixels > 100 );
	assert.ok( maxAngle < 0.025, `decoded normal max angle ${ maxAngle }` );
	result.normalComparison = { pixels, maxAngleRadians: maxAngle };

	for ( let cycle = 0; cycle < 3; cycle ++ ) {

		await configure( { ao: true, gi: true, aa: true, view: 'beauty', packing: 'rgb8' } );
		const restored = await inventory();
		result.checks.push( { name: `rich restored cycle ${ cycle }`, ...restored } );
		assert.equal( restored.memory.texturesSize, rich.memory.texturesSize );
		assert.equal( restored.memory.uniformBuffers, rich.memory.uniformBuffers );
		await configure( { ao: false, gi: false, aa: false } );
		const minimal = await inventory();
		assert.deepEqual( minimal.attachments, [ 'output' ] );
		assert.equal( minimal.memory.renderTargets, 1 );
		result.checks.push( { name: `minimal cycle ${ cycle }`, ...minimal } );

	}

	await page.screenshot( { path: path.join( artifacts, 'surface-minimal.png' ) } );
	await configure( { aa: true } );
	await settle();
	result.staticVelocity = await page.evaluate( async () => {

		const { renderer, state } = window.surfaceDemo;
		const { DataUtils } = await import( 'three/webgpu' );
		const index = state.source.renderTarget.textures.findIndex( t => t.name === 'motion' );
		const data = await renderer.readRenderTargetPixelsAsync( state.source.renderTarget, 500, 340, 64, 64, index );
		return Math.max( ...Array.from( data, value => Math.abs( DataUtils.fromHalfFloat( value ) ) ) );

	} );
	assert.ok( result.staticVelocity < 0.00001, 'static velocity excludes projection jitter' );

	// Attachment-only dependency: no depth sampling in the consumer's output.
	await configure( { aa: false } );
	await page.evaluate( async () => {

		const { pass } = await import( 'three/tsl' );
		const demo = window.surfaceDemo, source = demo.state.source;
		const consumer = pass( source.scene, source.camera, { samples: 0, depthTexture: source.getTexture( 'depth' ), autoClearDepth: false } ).before( source );
		window.depthReuse = consumer;
		window.orderEvents = [];
		for ( const [ name, node ] of [[ 'producer', source ], [ 'consumer', consumer ]] ) {

			const update = node.updateBefore;
			node.updateBefore = function ( frame ) {

				window.orderEvents.push( { name, frame: frame.frameId } );
				return update.call( this, frame );

			};

		}

		demo.pipeline.outputNode = consumer;
		demo.pipeline.needsUpdate = true;

	} );
	await settle();
	result.depthReuse = await page.evaluate( () => ( {
		sameDepth: window.depthReuse.renderTarget.depthTexture === window.surfaceDemo.state.source.renderTarget.depthTexture,
		clear: window.depthReuse.autoClearDepth,
		events: window.orderEvents
	} ) );
	assert.equal( result.depthReuse.sameDepth, true );
	assert.equal( result.depthReuse.clear, false );
	for ( let i = 0; i < result.depthReuse.events.length; i += 2 ) {

		assert.equal( result.depthReuse.events[ i ].name, 'producer' );
		assert.equal( result.depthReuse.events[ i + 1 ].name, 'consumer' );
		assert.equal( result.depthReuse.events[ i ].frame, result.depthReuse.events[ i + 1 ].frame );

	}

	await page.evaluate( () => window.depthReuse.dispose() );
	await configure( { ao: true, gi: true, aa: true, motion: true } );
	await page.setViewport( { width: 993, height: 677 } );
	await settle();
	result.checks.push( { name: 'motion + odd-size resize + history rebuild', ...await inventory() } );
	assert.deepEqual( result.checks.at( - 1 ).textures[ 0 ].size, { width: 993, height: 677, depthOrArrayLayers: 1 } );
	await page.evaluate( () => {

		window.surfaceDemo.camera.position.x += 2;
		window.surfaceDemo.rebuild();

	} );
	await settle();
	assert.deepEqual( result.errors, [] );
	console.log( 'PASS: GPU packing, plain/descriptor inventory, Inspector dedup, lifecycle, static motion, explicit depth ordering, resize/cut.' );

} catch ( error ) {

	result.errors.push( error.stack );
	console.error( error );
	process.exitCode = 1;

} finally {

	await writeFile( path.join( artifacts, 'surface-validation.json' ), JSON.stringify( result, null, 2 ) );
	await browser?.close();
	server.close();

}

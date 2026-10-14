import * as THREE from 'three';
import { NativeHDRLoader } from './loaders/NativeHDRLoader.js';

const status = document.querySelector( '#status' );
const format = document.querySelector( '#format' );
const exposure = document.querySelector( '#exposure' );
const file = document.querySelector( '#file' );
const backend = new URLSearchParams( location.search ).get( 'backend' ) || 'webgl';
const textures = {};
let customTexture;
let renderer, material, scene, camera;

window.nativeHDRTest = run().catch( ( error ) => {

	const result = { backend, error: error.message };
	status.textContent = JSON.stringify( result, null, 2 );
	return result;

} );

async function run() {

	const result = { backend, userAgent: navigator.userAgent, images: [] };

	if ( backend === 'webgpu' ) {

		const adapter = await navigator.gpu?.requestAdapter();
		if ( ! adapter ) throw new Error( 'No WebGPU adapter.' );
		const info = adapter.info;
		result.gpu = [ info.vendor, info.architecture, info.device, info.description ].join( ' ' );
		if ( adapter.isFallbackAdapter || info.isFallbackAdapter ) throw new Error( 'Software WebGPU adapter.' );
		const device = await adapter.requestDevice();
		const { WebGPURenderer } = await import( 'three/webgpu' );
		renderer = new WebGPURenderer( { device } );
		await renderer.init();

	} else {

		renderer = new THREE.WebGLRenderer();
		const gl = renderer.getContext();
		const extension = gl.getExtension( 'WEBGL_debug_renderer_info' );
		result.gpu = extension ? gl.getParameter( extension.UNMASKED_RENDERER_WEBGL ) : '';

	}

	if ( ! result.gpu.trim() || /swiftshader|llvmpipe|lavapipe|software|basic render/i.test( result.gpu ) ) {

		throw new Error( 'Hardware GPU identity unavailable: ' + result.gpu );

	}

	renderer.setSize( 800, 160 );
	document.body.insertBefore( renderer.domElement, status );
	scene = new THREE.Scene();
	camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 2 );
	camera.position.z = 1;
	material = new THREE.MeshBasicMaterial();
	scene.add( new THREE.Mesh( new THREE.PlaneGeometry( 2, 2 ), material ) );

	const manifest = await fetch( 'textures/hdr-native/neutral-pq.json' ).then( r => r.json() );
	const loader = new NativeHDRLoader();
	for ( const extension of [ 'avif', 'jxl' ] ) {

		const entry = { format: extension };
		result.images.push( entry );
		const url = 'textures/hdr-native/neutral-pq.' + extension;
		try {

			// Control: float16 canvas alone does not guarantee preservation of HDR.
			entry.defaultCanvas = await readDefaultCanvas( url, manifest );
			const texture = await loader.loadAsync( url );
			textures[ extension ] = texture;
			entry.cpu = samplePatches( texture.image.data, manifest );
			material.map = texture;
			material.toneMapped = false;
			material.needsUpdate = true;
			renderer.toneMapping = THREE.NoToneMapping;
			const target = new THREE.WebGLRenderTarget( manifest.width, manifest.height, { type: THREE.HalfFloatType, depthBuffer: false } );
			renderer.setRenderTarget( target );
			renderer.render( scene, camera );

			let data;
			if ( backend === 'webgpu' ) {

				data = await renderer.readRenderTargetPixelsAsync( target, 0, 0, manifest.width, manifest.height );

			} else {

				data = new Uint16Array( manifest.width * manifest.height * 4 );
				await renderer.readRenderTargetPixelsAsync( target, 0, 0, manifest.width, manifest.height, data );

			}

			entry.gpu = samplePatches( data, manifest );
			entry.expected = manifest.patches.map( patch => patch.linear );
			entry.maxRelativeError = Math.max( ...entry.gpu.slice( 1 ).map( ( value, i ) => Math.abs( value / entry.expected[ i + 1 ] - 1 ) ) );
			entry.passed = entry.cpu.every( Number.isFinite ) && entry.gpu.every( Number.isFinite ) &&
				Math.abs( entry.gpu[ 0 ] ) < 0.001 && entry.maxRelativeError < 0.02;
			renderer.setRenderTarget( null );
			target.dispose();

		} catch ( error ) {

			entry.error = error.message;

		}

	}

	format.onchange = preview;
	exposure.oninput = preview;
	file.onchange = async () => {

		if ( ! file.files[ 0 ] ) return;
		try {

			const texture = await loader.parse( await file.files[ 0 ].arrayBuffer() );
			if ( customTexture ) customTexture.dispose();
			customTexture = texture;
			preview();

		} catch ( error ) {

			status.textContent = error.message;

		}

	};

	preview();
	status.textContent = JSON.stringify( result, null, 2 );
	return result;

}

function samplePatches( data, manifest ) {

	return manifest.patches.map( ( { x, y } ) => THREE.DataUtils.fromHalfFloat( data[ ( y * manifest.width + x ) * 4 ] ) );

}

async function readDefaultCanvas( url, manifest ) {

	const image = new Image();
	image.src = url;
	await image.decode();
	const canvas = document.createElement( 'canvas' );
	canvas.width = image.naturalWidth;
	canvas.height = image.naturalHeight;
	const context = canvas.getContext( '2d', { colorSpace: 'srgb', colorType: 'float16' } );
	context.drawImage( image, 0, 0 );
	const pixels = context.getImageData( 0, 0, canvas.width, canvas.height, { colorSpace: 'srgb', pixelFormat: 'rgba-float16' } );
	return manifest.patches.map( ( { x, y } ) => pixels.data[ ( y * manifest.width + x ) * 4 ] );

}

function preview() {

	if ( ! renderer ) return;
	material.map = customTexture || textures[ format.value ] || null;
	material.toneMapped = true;
	material.needsUpdate = true;
	renderer.setRenderTarget( null );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = Math.pow( 2, Number( exposure.value ) );
	renderer.render( scene, camera );

}

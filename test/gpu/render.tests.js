import { afterAll, expect, test } from 'vitest';
import { Mesh, MeshBasicMaterial, OrthographicCamera, PlaneGeometry, Scene, WebGLRenderer } from '../../src/Three.js';
import { WebGPURenderer } from '../../src/Three.WebGPU.js';
import { assertHardwareRenderer } from './hardware.js';

const backend = process.env.THREE_GPU_BACKEND;
let renderer;
afterAll( async () => renderer?.dispose() );

test( 'renders geometry and reads completed GPU pixels', async () => {

	const canvas = document.createElement( 'canvas' );
	renderer = backend === 'webgl' ? new WebGLRenderer( { canvas } ) : new WebGPURenderer( { canvas } );
	if ( backend === 'webgpu' ) await renderer.init();
	await assertHardwareRenderer( renderer, backend );
	renderer.setSize( 8, 8, false );
	const scene = new Scene();
	const geometry = new PlaneGeometry( 2, 2 );
	const material = new MeshBasicMaterial( { color: 0xff0000 } );
	scene.add( new Mesh( geometry, material ) );
	const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
	camera.position.z = 1;
	try {

		if ( backend === 'webgpu' ) await renderer.renderAsync( scene, camera );
		else renderer.render( scene, camera );
		let pixel;
		if ( backend === 'webgl' ) {

			pixel = new Uint8Array( 4 );
			const gl = renderer.getContext();
			gl.readPixels( 4, 4, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel );

		} else {

			const { data } = await canvas.readPixels();
			pixel = data.slice( ( 4 * 8 + 4 ) * 4, ( 4 * 8 + 5 ) * 4 );

		}

		expect( Array.from( pixel ) ).toEqual( [ 255, 0, 0, 255 ] );

	} finally {

		geometry.dispose();
		material.dispose();

	}

} );

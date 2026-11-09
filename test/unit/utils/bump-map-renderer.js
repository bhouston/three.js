import { Scene, OrthographicCamera, PerspectiveCamera, PlaneGeometry, Mesh, InstancedMesh, BatchedMesh, Matrix4, MeshNormalMaterial, WebGLRenderer, WebGLRenderTarget, FloatType, RGBAFormat, DataTexture, LinearFilter } from '../../../src/Three.js';
import { MeshNormalNodeMaterial, WebGPURenderer } from '../../../src/Three.WebGPU.js';
import { normalView, vec4 } from '../../../src/Three.TSL.js';

export function heightTexture( axis = 'x' ) {

	const size = 64;
	const data = new Float32Array( size * size * 4 );

	for ( let y = 0; y < size; y ++ ) {

		for ( let x = 0; x < size; x ++ ) {

			const i = ( y * size + x ) * 4;
			data[ i ] = axis === 'x' ? ( x + 0.5 ) / size : axis === 'y' ? ( y + 0.5 ) / size : 0.5;
			data[ i + 3 ] = 1;

		}

	}

	const texture = new DataTexture( data, size, size, RGBAFormat, FloatType );
	texture.minFilter = texture.magFilter = LinearFilter;
	texture.needsUpdate = true;
	return texture;

}

export function normalTexture( x = 0, y = 0 ) {

	const texture = new DataTexture( new Float32Array( [ x * 0.5 + 0.5, y * 0.5 + 0.5, Math.sqrt( 1 - x * x - y * y ) * 0.5 + 0.5, 1 ] ), 1, 1, RGBAFormat, FloatType );
	texture.needsUpdate = true;
	return texture;

}

export async function createBumpMapRenderer( backend ) {

	const renderer = backend === 'legacy' ? new WebGLRenderer() : new WebGPURenderer( { forceWebGL: backend === 'webgl' } );
	if ( backend !== 'legacy' ) await renderer.init();
	renderer.setSize( 32, 32 );

	const target = new WebGLRenderTarget( 32, 32, { type: FloatType, depthBuffer: false } );
	const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 100 );
	camera.position.z = 5;

	return {
		async render( parameters, { geometry = new PlaneGeometry( 2, 2 ), matrix = new Matrix4(), kind = 'mesh', back = false, resolution = 32, distance = 5, perspective = false } = {} ) {

			const material = backend === 'legacy' ? new MeshNormalMaterial( parameters ) : new MeshNormalNodeMaterial( parameters );

			if ( backend !== 'legacy' ) {

				// Read the shading normal directly, avoiding the display color convention
				// of MeshNormalNodeMaterial when rendering to a linear target.
				material.fragmentNode = vec4( normalView.mul( 0.5 ).add( 0.5 ), 1 );

			}

			let mesh;
			if ( kind === 'instance' ) {

				mesh = new InstancedMesh( geometry, material, 1 );
				mesh.setMatrixAt( 0, matrix );
				mesh.instanceMatrix.needsUpdate = true;

			} else if ( kind === 'batch' ) {

				mesh = new BatchedMesh( 1, geometry.attributes.position.count, geometry.index.count, material );
				mesh.setMatrixAt( mesh.addInstance( mesh.addGeometry( geometry ) ), matrix );

			} else {

				mesh = new Mesh( geometry, material );
				mesh.matrixAutoUpdate = false;
				mesh.matrix.copy( matrix );

			}

			const activeCamera = perspective ? new PerspectiveCamera( 45, 1, 0.1, 100 ) : camera;
			activeCamera.position.z = back ? - distance : distance;
			activeCamera.lookAt( 0, 0, 0 );
			target.setSize( resolution, resolution );
			const scene = new Scene();
			scene.add( mesh );
			renderer.setRenderTarget( target );
			renderer.render( scene, activeCamera );
			const pixel = backend === 'legacy' ? new Float32Array( 4 ) : null;
			const result = backend === 'legacy' ? await renderer.readRenderTargetPixelsAsync( target, resolution / 2, resolution / 2, 1, 1, pixel ) : await renderer.readRenderTargetPixelsAsync( target, resolution / 2, resolution / 2, 1, 1 );
			material.dispose();
			geometry.dispose();
			if ( mesh.isBatchedMesh ) mesh.dispose();
			return Array.from( result ).slice( 0, 3 ).map( value => value * 2 - 1 );

		},
		dispose() {

			target.dispose();
			renderer.dispose();

		}
	};

}

export function assertNormal( assert, actual, expected, message ) {

	assert.ok( actual.every( ( value, i ) => Number.isFinite( value ) && Math.abs( value - expected[ i ] ) < 0.002 ), `${ message }: ${ actual } vs ${ expected }` );

}

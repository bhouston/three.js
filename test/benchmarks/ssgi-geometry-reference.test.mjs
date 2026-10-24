import test from 'node:test';
import assert from 'node:assert/strict';
import { URL } from 'node:url';
import { Scene, Mesh, PlaneGeometry, MeshBasicMaterial, PerspectiveCamera, FrontSide, BackSide, DoubleSide, BoxGeometry, InstancedMesh, Matrix4, Vector3 } from '../../src/Three.WebGPU.js';
import { ssgiGeometryReference } from './ssgi-geometry-reference.js';

const threeModule = new URL( '../../src/Three.WebGPU.js', import.meta.url ).href;
function fixture() {

	const model = new Scene();
	const material = new MeshBasicMaterial( { side: FrontSide } );
	model.add( new Mesh( new PlaneGeometry( 4, 4 ), material ) );
	const camera = new PerspectiveCamera( 50, 4 / 3, 0.1, 20 );
	camera.position.z = 5;
	camera.updateMatrixWorld();
	return { model, camera, material };

}

test( 'an isolated plane has no outgoing hemisphere geometry coverage', async () => {

	const { model, camera, material } = fixture();
	const result = await ssgiGeometryReference( { model, camera, width: 32, height: 24, radius: 2, gridWidth: 8, gridHeight: 6, rays: 64, threeModule } );
	assert.ok( result.valid.some( Boolean ) );
	assert.ok( result.coverage.every( value => value === 0 ) );
	assert.equal( material.side, FrontSide );

} );

test( 'triangle corner coverage agrees with unit screen radiance irradiance', async () => {

	const { model, camera, material } = fixture();
	const wall = new Mesh( new PlaneGeometry( 4, 4 ), material );
	wall.rotation.y = - Math.PI / 2;
	wall.position.set( 1, 0, 1 );
	model.add( wall );
	const radiance = new Float32Array( 32 * 24 * 4 ).fill( 1 );
	const result = await ssgiGeometryReference( { model, camera, width: 32, height: 24, radius: 2, gridWidth: 8, gridHeight: 6, rays: 64, radiance, threeModule } );
	assert.ok( result.coverage.some( value => value > 0 ) );
	for ( let i = 0; i < result.coverage.length; i ++ ) {

		assert.ok( Math.abs( result.irradiance[ i * 3 ] / Math.PI - result.coverage[ i ] ) < 1e-12 );

	}
	assert.equal( material.side, FrontSide );

} );


test( 'an inward BackSide room remains open to the primary camera rays', async () => {

	const model = new Scene();
	const material = new MeshBasicMaterial( { side: BackSide } );
	model.add( new Mesh( new BoxGeometry( 4, 4, 4 ), material ) );
	const camera = new PerspectiveCamera( 50, 4 / 3, 0.1, 20 );
	camera.position.z = 5;
	camera.updateMatrixWorld();
	const result = await ssgiGeometryReference( { model, camera, width: 64, height: 48, radius: 4, gridWidth: 8, gridHeight: 6, rays: 64, threeModule } );
	assert.ok( result.valid.some( Boolean ) );
	assert.ok( result.points.filter( Boolean ).every( point => point[ 2 ] < 2 - 1e-6 ), 'Primary rays skip the front wall culled by BackSide' );
	assert.ok( result.coverage.some( value => value > 0.1 ), 'Visible inward surfaces exchange screen-available transport' );
	assert.equal( material.side, BackSide, 'Secondary ray blocking restores the authored sidedness' );

} );

test( 'rendered source backfaces block coverage without emitting toward the receiver', async () => {

	const model = new Scene();
	const material = new MeshBasicMaterial( { side: DoubleSide } );
	const floor = new Mesh( new PlaneGeometry( 6, 6 ), material );
	floor.rotation.x = - Math.PI / 2;
	floor.position.y = -1;
	const wall = new Mesh( new PlaneGeometry( 2, 2 ), material );
	wall.rotation.y = Math.PI / 2;
	model.add( floor, wall );
	const camera = new PerspectiveCamera( 50, 4 / 3, 0.1, 30 );
	camera.position.set( 4, 3, 5 );
	camera.lookAt( 0, 0, 0 );
	camera.updateMatrixWorld();
	const options = { model, camera, width: 64, height: 48, radius: 4, gridWidth: 16, gridHeight: 12, rays: 64, radiance: new Float32Array( 64 * 48 * 4 ).fill( 1 ), threeModule };
	const darkBackfaces = await ssgiGeometryReference( { ...options, backfaceLighting: 0 } );
	const litBackfaces = await ssgiGeometryReference( { ...options, backfaceLighting: 1 } );
	assert.deepEqual( darkBackfaces.coverage, litBackfaces.coverage, 'Source sidedness changes emitted light, never geometric coverage' );
	assert.ok( darkBackfaces.irradiance.every( ( value, i ) => value <= litBackfaces.irradiance[ i ] + 1e-12 ) );
	assert.ok( darkBackfaces.irradiance.some( ( value, i ) => value < litBackfaces.irradiance[ i ] - 0.01 ), 'Camera-facing source radiance is not emitted through its back side' );
	assert.equal( material.side, DoubleSide );

} );


test( 'transformed instanced geometry agrees with equivalent mesh geometry', async () => {

	const make = instanced => {

		const { model, camera, material } = fixture();
		const transform = new Matrix4().makeRotationY( - Math.PI / 2 );
		transform.setPosition( 1, 0, 1 );
		const wall = instanced ? new InstancedMesh( new PlaneGeometry( 4, 4 ), material, 1 ) : new Mesh( new PlaneGeometry( 4, 4 ), material );
		if ( instanced ) {

			wall.setMatrixAt( 0, transform );

		} else {

			// Match the Float32 storage used by an instance matrix exactly.
			wall.matrix.fromArray( new Float32Array( transform.elements ) );
			wall.matrixAutoUpdate = false;

		}

		model.add( wall );
		model.position.set( 2, 3, -4 );
		model.rotation.y = 0.23;
		model.scale.set( 1.1, 0.9, 1.2 );
		model.updateMatrixWorld( true );
		camera.position.set( 0, 0, 5 ).applyMatrix4( model.matrixWorld );
		camera.lookAt( new Vector3().applyMatrix4( model.matrixWorld ) );
		camera.updateMatrixWorld();
		return { model, camera };

	};
	const options = { width: 32, height: 24, radius: 2, gridWidth: 8, gridHeight: 6, rays: 64, threeModule };
	const mesh = await ssgiGeometryReference( { ...make( false ), ...options } );
	const instanced = await ssgiGeometryReference( { ...make( true ), ...options } );
	assert.deepEqual( instanced.valid, mesh.valid );
	assert.deepEqual( instanced.coverage, mesh.coverage, 'Instance and parent transforms are applied to ray hits and shading normals consistently' );
	assert.ok( instanced.coverage.some( value => value > 0 ) );

} );

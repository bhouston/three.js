import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';
import { Mesh } from '@src/objects/Mesh.js';
import { Raycaster } from '@src/core/Raycaster.js';
import { PlaneGeometry } from '@src/geometries/PlaneGeometry.js';
import { BoxGeometry } from '@src/geometries/BoxGeometry.js';
import { MeshBasicMaterial } from '@src/materials/MeshBasicMaterial.js';
import { Vector2 } from '@src/math/Vector2.js';
import { Vector3 } from '@src/math/Vector3.js';
import { DoubleSide } from '@src/constants.js';
import { Material } from '@src/materials/Material.js';

describe( 'Objects', () => {

	describe( 'Mesh', () => {

		test( 'Extending', () => {

			const mesh = new Mesh();
			expect( mesh instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Mesh();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Mesh();
			expect( object.type === 'Mesh' ).toBeTruthy();

		} );

		test( 'isMesh', () => {

			const object = new Mesh();
			expect( object.isMesh ).toBeTruthy();

		} );

		test( 'copy/material', () => {

			// Material arrays are cloned
			const mesh1 = new Mesh();
			mesh1.material = [ new Material() ];

			const copy1 = mesh1.clone();
			expect( mesh1.material ).not.toBe( copy1.material );

			// Non arrays are not cloned
			const mesh2 = new Mesh();
			mesh1.material = new Material();
			const copy2 = mesh2.clone();
			expect( mesh2.material ).toBe( copy2.material );

		} );

		test.todo( 'raycast', () => {

			const geometry = new PlaneGeometry();
			const material = new MeshBasicMaterial();

			const mesh = new Mesh( geometry, material );

			const raycaster = new Raycaster();
			raycaster.ray.origin.set( 0.25, 0.25, 1 );
			raycaster.ray.direction.set( 0, 0, - 1 );

			const intersections = [];

			mesh.raycast( raycaster, intersections );

			const intersection = intersections[ 0 ];

			expect( intersection.object ).toBe( mesh );
			expect( intersection.distance ).toBe( 1 );
			expect( intersection.faceIndex ).toBe( 1 );
			expect( intersection.face ).toEqual( { a: 0, b: 2, c: 1 } );
			expect( intersection.point ).toEqual( new Vector3( 0.25, 0.25, 0 ) );
			expect( intersection.uv ).toEqual( new Vector2( 0.75, 0.75 ) );

		} );

		test( 'raycast/range', () => {

			const geometry = new BoxGeometry( 1, 1, 1 );
			const material = new MeshBasicMaterial( { side: DoubleSide } );
			const mesh = new Mesh( geometry, material );
			const raycaster = new Raycaster();
			const intersections = [];

			raycaster.ray.origin.set( 0, 0, 0 );
			raycaster.ray.direction.set( 1, 0, 0 );
			raycaster.near = 100;
			raycaster.far = 200;

			mesh.matrixWorld.identity();
			mesh.position.setX( 150 );
			mesh.updateMatrixWorld( true );
			intersections.length = 0;
			mesh.raycast( raycaster, intersections );
			expect( intersections.length > 0 ).toBeTruthy();

			mesh.matrixWorld.identity();
			mesh.position.setX( raycaster.near );
			mesh.updateMatrixWorld( true );
			intersections.length = 0;
			mesh.raycast( raycaster, intersections );
			expect( intersections.length > 0 ).toBeTruthy();

			mesh.matrixWorld.identity();
			mesh.position.setX( raycaster.far );
			mesh.updateMatrixWorld( true );
			intersections.length = 0;
			mesh.raycast( raycaster, intersections );
			expect( intersections.length > 0 ).toBeTruthy();

			mesh.matrixWorld.identity();
			mesh.position.setX( 150 );
			mesh.scale.setY( 9999 );
			mesh.updateMatrixWorld( true );
			intersections.length = 0;
			mesh.raycast( raycaster, intersections );
			expect( intersections.length > 0 ).toBeTruthy();

			mesh.matrixWorld.identity();
			mesh.position.setX( - 9999 );
			mesh.updateMatrixWorld( true );
			intersections.length = 0;
			mesh.raycast( raycaster, intersections );
			expect( intersections.length === 0 ).toBeTruthy();

			mesh.matrixWorld.identity();
			mesh.position.setX( 9999 );
			mesh.updateMatrixWorld( true );
			intersections.length = 0;
			mesh.raycast( raycaster, intersections );
			expect( intersections.length === 0 ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';

import { Raycaster } from '@src/core/Raycaster.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Mesh } from '@src/objects/Mesh.js';
import { SphereGeometry } from '@src/geometries/SphereGeometry.js';
import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { Line } from '@src/objects/Line.js';
import { Points } from '@src/objects/Points.js';
import { PerspectiveCamera } from '@src/cameras/PerspectiveCamera.js';
import { OrthographicCamera } from '@src/cameras/OrthographicCamera.js';
import { WebGPUCoordinateSystem } from '@src/constants.js';

function checkRayDirectionAgainstReferenceVector( rayDirection, refVector ) {

	expect( refVector.x - rayDirection.x <= Number.EPSILON && refVector.y - rayDirection.y <= Number.EPSILON && refVector.z - rayDirection.z <= Number.EPSILON ).toBeTruthy();

}

function getRaycaster() {

	return new Raycaster(
		new Vector3( 0, 0, 0 ),
		new Vector3( 0, 0, - 1 ),
		1,
		100
	);

}

function getObjectsToCheck() {

	const objects = [];

	const sphere1 = getSphere();
	sphere1.position.set( 0, 0, - 10 );
	sphere1.name = 1;
	objects.push( sphere1 );

	const sphere11 = getSphere();
	sphere11.position.set( 0, 0, 1 );
	sphere11.name = 11;
	sphere1.add( sphere11 );

	const sphere12 = getSphere();
	sphere12.position.set( 0, 0, - 1 );
	sphere12.name = 12;
	sphere1.add( sphere12 );

	const sphere2 = getSphere();
	sphere2.position.set( - 5, 0, - 5 );
	sphere2.name = 2;
	objects.push( sphere2 );

	for ( let i = 0; i < objects.length; i ++ ) {

		objects[ i ].updateMatrixWorld();

	}

	return objects;

}

function getSphere() {

	return new Mesh( new SphereGeometry( 1, 100, 100 ) );

}

describe( 'Core', () => {

	describe( 'Raycaster', () => {

		test( 'Instancing', () => {

			// no params
			const object = new Raycaster();
			expect( object ).toBeTruthy();

		} );

		test( 'set', () => {

			const origin = new Vector3( 0, 0, 0 );
			const direction = new Vector3( 0, 0, - 1 );
			const a = new Raycaster( origin.clone(), direction.clone() );

			expect( a.ray.origin ).toEqual( origin );
			expect( a.ray.direction ).toEqual( direction );

			origin.set( 1, 1, 1 );
			direction.set( - 1, 0, 0 );
			a.set( origin, direction );

			expect( a.ray.origin ).toEqual( origin );
			expect( a.ray.direction ).toEqual( direction );

		} );

		test( 'setFromCamera (Perspective)', () => {

			const raycaster = new Raycaster();
			const rayDirection = raycaster.ray.direction;
			const camera = new PerspectiveCamera( 90, 1, 1, 1000 );

			raycaster.setFromCamera( {
				x: 0,
				y: 0
			}, camera );
			expect( rayDirection.x === 0 && rayDirection.y === 0 && rayDirection.z === - 1 ).toBeTruthy();

			const step = 0.1;

			for ( let x = - 1; x <= 1; x += step ) {

				for ( let y = - 1; y <= 1; y += step ) {

					raycaster.setFromCamera( {
						x,
						y
					}, camera );

					const refVector = new Vector3( x, y, - 1 ).normalize();

					checkRayDirectionAgainstReferenceVector( rayDirection, refVector );

				}

			}

		} );

		test( 'setFromCamera (Orthographic)', () => {

			const raycaster = new Raycaster();
			const rayOrigin = raycaster.ray.origin;
			const rayDirection = raycaster.ray.direction;
			const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0, 1000 );
			const expectedOrigin = new Vector3( 0, 0, 0 );
			const expectedDirection = new Vector3( 0, 0, - 1 );

			raycaster.setFromCamera( {
				x: 0,
				y: 0
			}, camera );
			expect( rayOrigin ).toEqual( expectedOrigin );
			expect( rayDirection ).toEqual( expectedDirection );

		} );

		test( 'intersectObject', () => {

			const raycaster = getRaycaster();
			const objectsToCheck = getObjectsToCheck();

			expect( raycaster.intersectObject( objectsToCheck[ 0 ], false ).length === 1 ).toBeTruthy();

			expect( raycaster.intersectObject( objectsToCheck[ 0 ] ).length === 3 ).toBeTruthy();

			const intersections = raycaster.intersectObject( objectsToCheck[ 0 ] );
			for ( let i = 0; i < intersections.length - 1; i ++ ) {

				expect( intersections[ i ].distance <= intersections[ i + 1 ].distance ).toBeTruthy();

			}

		} );

		test( 'intersectObjects', () => {

			const raycaster = getRaycaster();
			const objectsToCheck = getObjectsToCheck();

			expect( raycaster.intersectObjects( objectsToCheck, false ).length === 1 ).toBeTruthy();

			expect( raycaster.intersectObjects( objectsToCheck ).length === 3 ).toBeTruthy();

			const intersections = raycaster.intersectObjects( objectsToCheck );
			for ( let i = 0; i < intersections.length - 1; i ++ ) {

				expect( intersections[ i ].distance <= intersections[ i + 1 ].distance ).toBeTruthy();

			}

		} );

		test( 'intersectObject (Perspective, WebGPU coordinate system)', () => {

			const camera = new PerspectiveCamera( 90, 1, 0.1, 100 );
			camera.coordinateSystem = WebGPUCoordinateSystem;
			camera.updateProjectionMatrix();

			const front = getSphere();
			front.position.set( 0, 0, - 5 );
			front.updateMatrixWorld();

			const behind = getSphere();
			behind.position.set( 0, 0, 5 );
			behind.updateMatrixWorld();

			const raycaster = new Raycaster();
			raycaster.setFromCamera( { x: 0, y: 0 }, camera );

			expect( raycaster.intersectObject( front ).length ).toBe( 1 );

			expect( raycaster.intersectObject( behind ).length ).toBe( 0 );

		} );

		test( 'intersectObject (Orthographic, WebGPU coordinate system)', () => {

			const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
			camera.coordinateSystem = WebGPUCoordinateSystem;
			camera.updateProjectionMatrix();

			const front = getSphere();
			front.position.set( 0, 0, - 5 );
			front.updateMatrixWorld();

			const behind = getSphere();
			behind.position.set( 0, 0, 5 );
			behind.updateMatrixWorld();

			const raycaster = new Raycaster();
			raycaster.setFromCamera( { x: 0, y: 0 }, camera );

			expect( raycaster.intersectObject( front ).length ).toBe( 1 );

			expect( raycaster.intersectObject( behind ).length ).toBe( 0 );

		} );

		test( 'intersectObject (Perspective, reversed depth)', () => {

			const camera = new PerspectiveCamera( 90, 1, 0.1, 100 );
			camera._reversedDepth = true;
			camera.updateProjectionMatrix();

			const front = getSphere();
			front.position.set( 0, 0, - 5 );
			front.updateMatrixWorld();

			const behind = getSphere();
			behind.position.set( 0, 0, 5 );
			behind.updateMatrixWorld();

			const raycaster = new Raycaster();
			raycaster.setFromCamera( { x: 0, y: 0 }, camera );

			expect( raycaster.intersectObject( front ).length ).toBe( 1 );

			expect( raycaster.intersectObject( behind ).length ).toBe( 0 );

		} );

		test( 'intersectObject (Orthographic, reversed depth)', () => {

			const camera = new OrthographicCamera( - 1, 1, 1, - 1, 0.1, 10 );
			camera._reversedDepth = true;
			camera.updateProjectionMatrix();

			const front = getSphere();
			front.position.set( 0, 0, - 5 );
			front.updateMatrixWorld();

			const behind = getSphere();
			behind.position.set( 0, 0, 5 );
			behind.updateMatrixWorld();

			const raycaster = new Raycaster();
			raycaster.setFromCamera( { x: 0, y: 0 }, camera );

			expect( raycaster.intersectObject( front ).length ).toBe( 1 );

			expect( raycaster.intersectObject( behind ).length ).toBe( 0 );

		} );

		test( 'Line intersection threshold', () => {

			const raycaster = getRaycaster();
			const points = [ new Vector3( - 2, - 10, - 5 ), new Vector3( - 2, 10, - 5 ) ];
			const geometry = new BufferGeometry().setFromPoints( points );
			const line = new Line( geometry, null );

			raycaster.params.Line.threshold = 1.999;
			expect( raycaster.intersectObject( line ).length === 0 ).toBeTruthy();

			raycaster.params.Line.threshold = 2.001;
			expect( raycaster.intersectObject( line ).length === 1 ).toBeTruthy();

		} );

		test( 'Points intersection threshold', () => {

			const raycaster = getRaycaster();
			const coordinates = [ new Vector3( - 2, 0, - 5 ) ];
			const geometry = new BufferGeometry().setFromPoints( coordinates );
			const points = new Points( geometry, null );

			raycaster.params.Points.threshold = 1.999;
			expect( raycaster.intersectObject( points ).length === 0 ).toBeTruthy();

			raycaster.params.Points.threshold = 2.001;
			expect( raycaster.intersectObject( points ).length === 1 ).toBeTruthy();

		} );

	} );

} );

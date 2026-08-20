import { describe, test, expect } from 'vitest';
import { Frustum } from '@src/math/Frustum.js';

import { Sphere } from '@src/math/Sphere.js';
import { Plane } from '@src/math/Plane.js';
import { Sprite } from '@src/objects/Sprite.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import { Box3 } from '@src/math/Box3.js';
import { Mesh } from '@src/objects/Mesh.js';
import { BoxGeometry } from '@src/geometries/BoxGeometry.js';
import { zero3, one3, eps } from '@test-utils/math-constants.js';

const unit3 = new Vector3( 1, 0, 0 );

describe( 'Maths', () => {

	describe( 'Frustum', () => {

		test( 'Instancing', () => {

			let a = new Frustum();

			expect( a.planes !== undefined ).toBeTruthy();
			expect( a.planes.length === 6 ).toBeTruthy();

			const pDefault = new Plane();
			for ( let i = 0; i < 6; i ++ ) {

				expect( a.planes[ i ].equals( pDefault ) ).toBeTruthy();

			}

			const p0 = new Plane( unit3, - 1 );
			const p1 = new Plane( unit3, 1 );
			const p2 = new Plane( unit3, 2 );
			const p3 = new Plane( unit3, 3 );
			const p4 = new Plane( unit3, 4 );
			const p5 = new Plane( unit3, 5 );

			a = new Frustum( p0, p1, p2, p3, p4, p5 );
			expect( a.planes[ 0 ].equals( p0 ) ).toBeTruthy();
			expect( a.planes[ 1 ].equals( p1 ) ).toBeTruthy();
			expect( a.planes[ 2 ].equals( p2 ) ).toBeTruthy();
			expect( a.planes[ 3 ].equals( p3 ) ).toBeTruthy();
			expect( a.planes[ 4 ].equals( p4 ) ).toBeTruthy();
			expect( a.planes[ 5 ].equals( p5 ) ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Frustum();
			const p0 = new Plane( unit3, - 1 );
			const p1 = new Plane( unit3, 1 );
			const p2 = new Plane( unit3, 2 );
			const p3 = new Plane( unit3, 3 );
			const p4 = new Plane( unit3, 4 );
			const p5 = new Plane( unit3, 5 );

			a.set( p0, p1, p2, p3, p4, p5 );

			expect( a.planes[ 0 ].equals( p0 ) ).toBeTruthy();
			expect( a.planes[ 1 ].equals( p1 ) ).toBeTruthy();
			expect( a.planes[ 2 ].equals( p2 ) ).toBeTruthy();
			expect( a.planes[ 3 ].equals( p3 ) ).toBeTruthy();
			expect( a.planes[ 4 ].equals( p4 ) ).toBeTruthy();
			expect( a.planes[ 5 ].equals( p5 ) ).toBeTruthy();

		} );

		test( 'clone', () => {

			const p0 = new Plane( unit3, - 1 );
			const p1 = new Plane( unit3, 1 );
			const p2 = new Plane( unit3, 2 );
			const p3 = new Plane( unit3, 3 );
			const p4 = new Plane( unit3, 4 );
			const p5 = new Plane( unit3, 5 );

			const b = new Frustum( p0, p1, p2, p3, p4, p5 );
			const a = b.clone();
			expect( a.planes[ 0 ].equals( p0 ) ).toBeTruthy();
			expect( a.planes[ 1 ].equals( p1 ) ).toBeTruthy();
			expect( a.planes[ 2 ].equals( p2 ) ).toBeTruthy();
			expect( a.planes[ 3 ].equals( p3 ) ).toBeTruthy();
			expect( a.planes[ 4 ].equals( p4 ) ).toBeTruthy();
			expect( a.planes[ 5 ].equals( p5 ) ).toBeTruthy();

			// ensure it is a true copy by modifying source
			a.planes[ 0 ].copy( p1 );
			expect( b.planes[ 0 ].equals( p0 ) ).toBeTruthy();

		} );

		test( 'copy', () => {

			const p0 = new Plane( unit3, - 1 );
			const p1 = new Plane( unit3, 1 );
			const p2 = new Plane( unit3, 2 );
			const p3 = new Plane( unit3, 3 );
			const p4 = new Plane( unit3, 4 );
			const p5 = new Plane( unit3, 5 );

			const b = new Frustum( p0, p1, p2, p3, p4, p5 );
			const a = new Frustum().copy( b );
			expect( a.planes[ 0 ].equals( p0 ) ).toBeTruthy();
			expect( a.planes[ 1 ].equals( p1 ) ).toBeTruthy();
			expect( a.planes[ 2 ].equals( p2 ) ).toBeTruthy();
			expect( a.planes[ 3 ].equals( p3 ) ).toBeTruthy();
			expect( a.planes[ 4 ].equals( p4 ) ).toBeTruthy();
			expect( a.planes[ 5 ].equals( p5 ) ).toBeTruthy();

			// ensure it is a true copy by modifying source
			b.planes[ 0 ] = p1;
			expect( a.planes[ 0 ].equals( p0 ) ).toBeTruthy();

		} );

		test( 'setFromProjectionMatrix/makeOrthographic/containsPoint', () => {

			const m = new Matrix4().makeOrthographic( - 1, 1, 1, - 1, 1, 100 );
			const a = new Frustum().setFromProjectionMatrix( m );

			expect( ! a.containsPoint( new Vector3( 0, 0, 0 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 0, 0, - 50 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 0, 0, - 1.001 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( - 1, - 1, - 1.001 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( - 1.1, - 1.1, - 1.001 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 1, 1, - 1.001 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( 1.1, 1.1, - 1.001 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 0, 0, - 99.999 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( - 0.999, - 0.999, - 99.999 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( - 1.1, - 1.1, - 100.1 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 0.999, 0.999, - 99.999 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( 1.1, 1.1, - 100.1 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( 0, 0, - 101 ) ) ).toBeTruthy();

		} );

		test( 'setFromProjectionMatrix/makePerspective/containsPoint', () => {

			const m = new Matrix4().makePerspective( - 1, 1, 1, - 1, 1, 100 );
			const a = new Frustum().setFromProjectionMatrix( m );

			expect( ! a.containsPoint( new Vector3( 0, 0, 0 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 0, 0, - 50 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 0, 0, - 1.001 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( - 1, - 1, - 1.001 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( - 1.1, - 1.1, - 1.001 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 1, 1, - 1.001 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( 1.1, 1.1, - 1.001 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 0, 0, - 99.999 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( - 99.999, - 99.999, - 99.999 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( - 100.1, - 100.1, - 100.1 ) ) ).toBeTruthy();
			expect( a.containsPoint( new Vector3( 99.999, 99.999, - 99.999 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( 100.1, 100.1, - 100.1 ) ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( 0, 0, - 101 ) ) ).toBeTruthy();

		} );

		test( 'setFromProjectionMatrix/makePerspective/intersectsSphere', () => {

			const m = new Matrix4().makePerspective( - 1, 1, 1, - 1, 1, 100 );
			const a = new Frustum().setFromProjectionMatrix( m );

			expect( ! a.intersectsSphere( new Sphere( new Vector3( 0, 0, 0 ), 0 ) ) ).toBeTruthy();
			expect( ! a.intersectsSphere( new Sphere( new Vector3( 0, 0, 0 ), 0.9 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 0, 0, 0 ), 1.1 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 0, 0, - 50 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 0, 0, - 1.001 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( - 1, - 1, - 1.001 ), 0 ) ) ).toBeTruthy();
			expect( ! a.intersectsSphere( new Sphere( new Vector3( - 1.1, - 1.1, - 1.001 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( - 1.1, - 1.1, - 1.001 ), 0.5 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 1, 1, - 1.001 ), 0 ) ) ).toBeTruthy();
			expect( ! a.intersectsSphere( new Sphere( new Vector3( 1.1, 1.1, - 1.001 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 1.1, 1.1, - 1.001 ), 0.5 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 0, 0, - 99.999 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( - 99.999, - 99.999, - 99.999 ), 0 ) ) ).toBeTruthy();
			expect( ! a.intersectsSphere( new Sphere( new Vector3( - 100.1, - 100.1, - 100.1 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( - 100.1, - 100.1, - 100.1 ), 0.5 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 99.999, 99.999, - 99.999 ), 0 ) ) ).toBeTruthy();
			expect( ! a.intersectsSphere( new Sphere( new Vector3( 100.1, 100.1, - 100.1 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 100.1, 100.1, - 100.1 ), 0.2 ) ) ).toBeTruthy();
			expect( ! a.intersectsSphere( new Sphere( new Vector3( 0, 0, - 101 ), 0 ) ) ).toBeTruthy();
			expect( a.intersectsSphere( new Sphere( new Vector3( 0, 0, - 101 ), 1.1 ) ) ).toBeTruthy();

		} );

		test( 'intersectsObject', () => {

			const m = new Matrix4().makePerspective( - 1, 1, 1, - 1, 1, 100 );
			const a = new Frustum().setFromProjectionMatrix( m );
			const object = new Mesh( new BoxGeometry( 1, 1, 1 ) );
			let intersects;

			intersects = a.intersectsObject( object );
			expect( intersects ).toBeFalsy();

			object.position.set( - 1, - 1, - 1 );
			object.updateMatrixWorld();

			intersects = a.intersectsObject( object );
			expect( intersects ).toBeTruthy();

			object.position.set( 1, 1, 1 );
			object.updateMatrixWorld();

			intersects = a.intersectsObject( object );
			expect( intersects ).toBeFalsy();

		} );

		test( 'intersectsSprite', () => {

			const m = new Matrix4().makePerspective( - 1, 1, 1, - 1, 1, 100 );
			const a = new Frustum().setFromProjectionMatrix( m );
			const sprite = new Sprite();
			let intersects;

			intersects = a.intersectsSprite( sprite );
			expect( intersects ).toBeFalsy();

			sprite.position.set( - 1, - 1, - 1 );
			sprite.updateMatrixWorld();

			intersects = a.intersectsSprite( sprite );
			expect( intersects ).toBeTruthy();

		} );

		test( 'intersectsBox', () => {

			const m = new Matrix4().makePerspective( - 1, 1, 1, - 1, 1, 100 );
			const a = new Frustum().setFromProjectionMatrix( m );
			const box = new Box3( zero3.clone(), one3.clone() );
			let intersects;

			intersects = a.intersectsBox( box );
			expect( intersects ).toBeFalsy();

			// add eps so that we prevent box touching the frustum,
			// which might intersect depending on floating point numerics
			box.translate( new Vector3( - 1 - eps, - 1 - eps, - 1 - eps ) );

			intersects = a.intersectsBox( box );
			expect( intersects ).toBeTruthy();

		} );

	} );

} );

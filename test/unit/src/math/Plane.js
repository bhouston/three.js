import { describe, test, expect } from 'vitest';
import { Plane } from '@src/math/Plane.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Line3 } from '@src/math/Line3.js';
import { Sphere } from '@src/math/Sphere.js';
import { Box3 } from '@src/math/Box3.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import {
	x,
	y,
	z,
	w,
	zero3,
	one3
} from '@test-utils/math-constants.js';

function comparePlane( a, b, threshold ) {

	threshold = threshold || 0.0001;
	return ( a.normal.distanceTo( b.normal ) < threshold &&
	Math.abs( a.constant - b.constant ) < threshold );

}

describe( 'Maths', () => {

	describe( 'Plane', () => {

		test( 'Instancing', () => {

			let a = new Plane();
			expect( a.normal.x == 1 ).toBeTruthy();
			expect( a.normal.y == 0 ).toBeTruthy();
			expect( a.normal.z == 0 ).toBeTruthy();
			expect( a.constant == 0 ).toBeTruthy();

			a = new Plane( one3.clone(), 0 );
			expect( a.normal.x == 1 ).toBeTruthy();
			expect( a.normal.y == 1 ).toBeTruthy();
			expect( a.normal.z == 1 ).toBeTruthy();
			expect( a.constant == 0 ).toBeTruthy();

			a = new Plane( one3.clone(), 1 );
			expect( a.normal.x == 1 ).toBeTruthy();
			expect( a.normal.y == 1 ).toBeTruthy();
			expect( a.normal.z == 1 ).toBeTruthy();
			expect( a.constant == 1 ).toBeTruthy();

		} );

		test( 'isPlane', () => {

			const a = new Plane();
			expect( a.isPlane === true ).toBeTruthy();

			const b = new Vector3();
			expect( ! b.isPlane ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Plane();
			expect( a.normal.x == 1 ).toBeTruthy();
			expect( a.normal.y == 0 ).toBeTruthy();
			expect( a.normal.z == 0 ).toBeTruthy();
			expect( a.constant == 0 ).toBeTruthy();

			const b = a.clone().set( new Vector3( x, y, z ), w );
			expect( b.normal.x == x ).toBeTruthy();
			expect( b.normal.y == y ).toBeTruthy();
			expect( b.normal.z == z ).toBeTruthy();
			expect( b.constant == w ).toBeTruthy();

		} );

		test( 'setComponents', () => {

			const a = new Plane();
			expect( a.normal.x == 1 ).toBeTruthy();
			expect( a.normal.y == 0 ).toBeTruthy();
			expect( a.normal.z == 0 ).toBeTruthy();
			expect( a.constant == 0 ).toBeTruthy();

			const b = a.clone().setComponents( x, y, z, w );
			expect( b.normal.x == x ).toBeTruthy();
			expect( b.normal.y == y ).toBeTruthy();
			expect( b.normal.z == z ).toBeTruthy();
			expect( b.constant == w ).toBeTruthy();

		} );

		test( 'setFromNormalAndCoplanarPoint', () => {

			const normal = one3.clone().normalize();
			const a = new Plane().setFromNormalAndCoplanarPoint( normal, zero3 );

			expect( a.normal.equals( normal ) ).toBeTruthy();
			expect( a.constant == 0 ).toBeTruthy();

		} );

		test( 'setFromCoplanarPoints', () => {

			const a = new Plane();
			const v1 = new Vector3( 2.0, 0.5, 0.25 );
			const v2 = new Vector3( 2.0, - 0.5, 1.25 );
			const v3 = new Vector3( 2.0, - 3.5, 2.2 );
			const normal = new Vector3( 1, 0, 0 );
			const constant = - 2;

			a.setFromCoplanarPoints( v1, v2, v3 );

			expect( a.normal.equals( normal ) ).toBeTruthy();
			expect( a.constant ).toBe( constant );

		} );

		test( 'clone', () => {

			const a = new Plane( new Vector3( 2.0, 0.5, 0.25 ) );
			const b = a.clone();

			expect( a.equals( b ) ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Plane( new Vector3( x, y, z ), w );
			const b = new Plane().copy( a );
			expect( b.normal.x == x ).toBeTruthy();
			expect( b.normal.y == y ).toBeTruthy();
			expect( b.normal.z == z ).toBeTruthy();
			expect( b.constant == w ).toBeTruthy();

			// ensure that it is a true copy
			a.normal.x = 0;
			a.normal.y = - 1;
			a.normal.z = - 2;
			a.constant = - 3;
			expect( b.normal.x == x ).toBeTruthy();
			expect( b.normal.y == y ).toBeTruthy();
			expect( b.normal.z == z ).toBeTruthy();
			expect( b.constant == w ).toBeTruthy();

		} );

		test( 'normalize', () => {

			const a = new Plane( new Vector3( 2, 0, 0 ), 2 );

			a.normalize();
			expect( a.normal.length() == 1 ).toBeTruthy();
			expect( a.normal.equals( new Vector3( 1, 0, 0 ) ) ).toBeTruthy();
			expect( a.constant == 1 ).toBeTruthy();

		} );

		test( 'negate/distanceToPoint', () => {

			const a = new Plane( new Vector3( 2, 0, 0 ), - 2 );

			a.normalize();
			expect( a.distanceToPoint( new Vector3( 4, 0, 0 ) ) === 3 ).toBeTruthy();
			expect( a.distanceToPoint( new Vector3( 1, 0, 0 ) ) === 0 ).toBeTruthy();

			a.negate();
			expect( a.distanceToPoint( new Vector3( 4, 0, 0 ) ) === - 3 ).toBeTruthy();
			expect( a.distanceToPoint( new Vector3( 1, 0, 0 ) ) === 0 ).toBeTruthy();

		} );

		test( 'distanceToPoint', () => {

			const a = new Plane( new Vector3( 2, 0, 0 ), - 2 );
			const point = new Vector3();

			a.normalize().projectPoint( zero3.clone(), point );
			expect( a.distanceToPoint( point ) === 0 ).toBeTruthy();
			expect( a.distanceToPoint( new Vector3( 4, 0, 0 ) ) === 3 ).toBeTruthy();

		} );

		test( 'distanceToSphere', () => {

			const a = new Plane( new Vector3( 1, 0, 0 ), 0 );

			const b = new Sphere( new Vector3( 2, 0, 0 ), 1 );

			expect( a.distanceToSphere( b ) === 1 ).toBeTruthy();

			a.set( new Vector3( 1, 0, 0 ), 2 );
			expect( a.distanceToSphere( b ) === 3 ).toBeTruthy();
			a.set( new Vector3( 1, 0, 0 ), - 2 );
			expect( a.distanceToSphere( b ) === - 1 ).toBeTruthy();

		} );

		test( 'projectPoint', () => {

			let a = new Plane( new Vector3( 1, 0, 0 ), 0 );
			const point = new Vector3();

			a.projectPoint( new Vector3( 10, 0, 0 ), point );
			expect( point.equals( zero3 ) ).toBeTruthy();
			a.projectPoint( new Vector3( - 10, 0, 0 ), point );
			expect( point.equals( zero3 ) ).toBeTruthy();

			a = new Plane( new Vector3( 0, 1, 0 ), - 1 );
			a.projectPoint( new Vector3( 0, 0, 0 ), point );
			expect( point.equals( new Vector3( 0, 1, 0 ) ) ).toBeTruthy();
			a.projectPoint( new Vector3( 0, 1, 0 ), point );
			expect( point.equals( new Vector3( 0, 1, 0 ) ) ).toBeTruthy();

		} );

		test( 'intersectLine', () => {

			let a = new Plane( new Vector3( 1, 0, 0 ), 0 );
			const point = new Vector3();

			const l1 = new Line3( new Vector3( - 10, 0, 0 ), new Vector3( 10, 0, 0 ) );
			a.intersectLine( l1, point );
			expect( point.equals( new Vector3( 0, 0, 0 ) ) ).toBeTruthy();

			a = new Plane( new Vector3( 1, 0, 0 ), - 3 );
			a.intersectLine( l1, point );
			expect( point.equals( new Vector3( 3, 0, 0 ) ) ).toBeTruthy();

			// plane lies outside the segment's endpoints
			a = new Plane( new Vector3( 1, 0, 0 ), - 20 );
			const l2 = new Line3( new Vector3( - 10, 0, 0 ), new Vector3( 10, 0, 0 ) );

			expect( a.intersectLine( l2, point ) ).toBe( null );
			expect( a.intersectLine( l2, point, true ) ).toBe( null );

			const result = a.intersectLine( l2, point, false );
			expect( result === point ).toBeTruthy();
			expect( point.equals( new Vector3( 20, 0, 0 ) ) ).toBeTruthy();

		} );

		test( 'intersectsBox', () => {

			const a = new Box3( zero3.clone(), one3.clone() );
			const b = new Plane( new Vector3( 0, 1, 0 ), 1 );
			const c = new Plane( new Vector3( 0, 1, 0 ), 1.25 );
			const d = new Plane( new Vector3( 0, - 1, 0 ), 1.25 );
			const e = new Plane( new Vector3( 0, 1, 0 ), 0.25 );
			const f = new Plane( new Vector3( 0, 1, 0 ), - 0.25 );
			const g = new Plane( new Vector3( 0, 1, 0 ), - 0.75 );
			const h = new Plane( new Vector3( 0, 1, 0 ), - 1 );
			const i = new Plane( new Vector3( 1, 1, 1 ).normalize(), - 1.732 );
			const j = new Plane( new Vector3( 1, 1, 1 ).normalize(), - 1.733 );

			expect( ! b.intersectsBox( a ) ).toBeTruthy();
			expect( ! c.intersectsBox( a ) ).toBeTruthy();
			expect( ! d.intersectsBox( a ) ).toBeTruthy();
			expect( ! e.intersectsBox( a ) ).toBeTruthy();
			expect( f.intersectsBox( a ) ).toBeTruthy();
			expect( g.intersectsBox( a ) ).toBeTruthy();
			expect( h.intersectsBox( a ) ).toBeTruthy();
			expect( i.intersectsBox( a ) ).toBeTruthy();
			expect( ! j.intersectsBox( a ) ).toBeTruthy();

		} );

		test( 'intersectsSphere', () => {

			const a = new Sphere( zero3.clone(), 1 );
			const b = new Plane( new Vector3( 0, 1, 0 ), 1 );
			const c = new Plane( new Vector3( 0, 1, 0 ), 1.25 );
			const d = new Plane( new Vector3( 0, - 1, 0 ), 1.25 );

			expect( b.intersectsSphere( a ) ).toBeTruthy();
			expect( ! c.intersectsSphere( a ) ).toBeTruthy();
			expect( ! d.intersectsSphere( a ) ).toBeTruthy();

		} );

		test( 'coplanarPoint', () => {

			const point = new Vector3();

			let a = new Plane( new Vector3( 1, 0, 0 ), 0 );
			a.coplanarPoint( point );
			expect( a.distanceToPoint( point ) === 0 ).toBeTruthy();

			a = new Plane( new Vector3( 0, 1, 0 ), - 1 );
			a.coplanarPoint( point );
			expect( a.distanceToPoint( point ) === 0 ).toBeTruthy();

		} );

		test( 'applyMatrix4/translate', () => {

			let a = new Plane( new Vector3( 1, 0, 0 ), 0 );

			const m = new Matrix4();
			m.makeRotationZ( Math.PI * 0.5 );

			expect( comparePlane( a.clone().applyMatrix4( m ), new Plane( new Vector3( 0, 1, 0 ), 0 ) ) ).toBeTruthy();

			a = new Plane( new Vector3( 0, 1, 0 ), - 1 );
			expect( comparePlane( a.clone().applyMatrix4( m ), new Plane( new Vector3( - 1, 0, 0 ), - 1 ) ) ).toBeTruthy();

			m.makeTranslation( 1, 1, 1 );
			expect( comparePlane( a.clone().applyMatrix4( m ), a.clone().translate( new Vector3( 1, 1, 1 ) ) ) ).toBeTruthy();

		} );

		test( 'equals', () => {

			const a = new Plane( new Vector3( 1, 0, 0 ), 0 );
			const b = new Plane( new Vector3( 1, 0, 0 ), 1 );
			const c = new Plane( new Vector3( 0, 1, 0 ), 0 );

			expect( a.normal.equals( b.normal ) ).toBeTruthy();
			expect( a.normal.equals( c.normal ) ).toBeFalsy();

			expect( a.constant ).not.toBe( b.constant );
			expect( a.constant ).toBe( c.constant );

			expect( a.equals( b ) ).toBeFalsy();
			expect( a.equals( c ) ).toBeFalsy();

			a.copy( b );
			expect( a.normal.equals( b.normal ) ).toBeTruthy();
			expect( a.constant ).toBe( b.constant );
			expect( a.equals( b ) ).toBeTruthy();

		} );

	} );

} );

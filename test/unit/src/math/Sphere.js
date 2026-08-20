import { describe, test, expect } from 'vitest';
import { Box3 } from '@src/math/Box3.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Sphere } from '@src/math/Sphere.js';
import { Plane } from '@src/math/Plane.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import {
	zero3,
	one3,
	two3,
	eps
} from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Sphere', () => {

		test( 'Instancing', () => {

			let a = new Sphere();
			expect( a.center.equals( zero3 ) ).toBeTruthy();
			expect( a.radius == - 1 ).toBeTruthy();

			a = new Sphere( one3.clone(), 1 );
			expect( a.center.equals( one3 ) ).toBeTruthy();
			expect( a.radius == 1 ).toBeTruthy();

		} );

		test( 'isSphere', () => {

			const a = new Sphere();
			expect( a.isSphere === true ).toBeTruthy();

			const b = new Box3();
			expect( ! b.isSphere ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Sphere();
			expect( a.center.equals( zero3 ) ).toBeTruthy();
			expect( a.radius == - 1 ).toBeTruthy();

			a.set( one3, 1 );
			expect( a.center.equals( one3 ) ).toBeTruthy();
			expect( a.radius == 1 ).toBeTruthy();

		} );

		test( 'setFromPoints', () => {

			const a = new Sphere();
			const expectedCenter = new Vector3( 0.9330126941204071, 0, 0 );
			let expectedRadius = 1.3676668773461689;
			const optionalCenter = new Vector3( 1, 1, 1 );
			const points = [
				new Vector3( 1, 1, 0 ), new Vector3( 1, 1, 0 ),
				new Vector3( 1, 1, 0 ), new Vector3( 1, 1, 0 ),
				new Vector3( 1, 1, 0 ), new Vector3( 0.8660253882408142, 0.5, 0 ),
				new Vector3( - 0, 0.5, 0.8660253882408142 ), new Vector3( 1.8660253882408142, 0.5, 0 ),
				new Vector3( 0, 0.5, - 0.8660253882408142 ), new Vector3( 0.8660253882408142, 0.5, - 0 ),
				new Vector3( 0.8660253882408142, - 0.5, 0 ), new Vector3( - 0, - 0.5, 0.8660253882408142 ),
				new Vector3( 1.8660253882408142, - 0.5, 0 ), new Vector3( 0, - 0.5, - 0.8660253882408142 ),
				new Vector3( 0.8660253882408142, - 0.5, - 0 ), new Vector3( - 0, - 1, 0 ),
				new Vector3( - 0, - 1, 0 ), new Vector3( 0, - 1, 0 ),
				new Vector3( 0, - 1, - 0 ), new Vector3( - 0, - 1, - 0 ),
			];

			a.setFromPoints( points );
			expect( Math.abs( a.center.x - expectedCenter.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.center.y - expectedCenter.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.center.z - expectedCenter.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.radius - expectedRadius ) <= eps ).toBeTruthy();

			expectedRadius = 2.5946195770400102;
			a.setFromPoints( points, optionalCenter );
			expect( Math.abs( a.center.x - optionalCenter.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.center.y - optionalCenter.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.center.z - optionalCenter.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.radius - expectedRadius ) <= eps ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Sphere( one3.clone(), 1 );
			const b = new Sphere().copy( a );

			expect( b.center.equals( one3 ) ).toBeTruthy();
			expect( b.radius == 1 ).toBeTruthy();

			// ensure that it is a true copy
			a.center = zero3;
			a.radius = 0;
			expect( b.center.equals( one3 ) ).toBeTruthy();
			expect( b.radius == 1 ).toBeTruthy();

		} );

		test( 'isEmpty', () => {

			const a = new Sphere();
			expect( a.isEmpty() ).toBeTruthy();

			a.set( one3, 1 );
			expect( ! a.isEmpty() ).toBeTruthy();

			// Negative radius contains no points
			a.set( one3, - 1 );
			expect( a.isEmpty() ).toBeTruthy();

			// Zero radius contains only the center point
			a.set( one3, 0 );
			expect( ! a.isEmpty() ).toBeTruthy();

		} );

		test( 'makeEmpty', () => {

			const a = new Sphere( one3.clone(), 1 );

			expect( ! a.isEmpty() ).toBeTruthy();

			a.makeEmpty();
			expect( a.isEmpty() ).toBeTruthy();
			expect( a.center.equals( zero3 ) ).toBeTruthy();

		} );

		test( 'containsPoint', () => {

			const a = new Sphere( one3.clone(), 1 );

			expect( ! a.containsPoint( zero3 ) ).toBeTruthy();
			expect( a.containsPoint( one3 ) ).toBeTruthy();

			a.set( zero3, 0 );
			expect( a.containsPoint( a.center ) ).toBeTruthy();

		} );

		test( 'distanceToPoint', () => {

			const a = new Sphere( one3.clone(), 1 );

			expect( ( a.distanceToPoint( zero3 ) - 0.7320 ) < 0.001 ).toBeTruthy();
			expect( a.distanceToPoint( one3 ) === - 1 ).toBeTruthy();

		} );

		test( 'intersectsSphere', () => {

			const a = new Sphere( one3.clone(), 1 );
			const b = new Sphere( zero3.clone(), 1 );
			const c = new Sphere( zero3.clone(), 0.25 );

			expect( a.intersectsSphere( b ) ).toBeTruthy();
			expect( ! a.intersectsSphere( c ) ).toBeTruthy();

		} );

		test( 'intersectsBox', () => {

			const a = new Sphere( zero3, 1 );
			const b = new Sphere( new Vector3( - 5, - 5, - 5 ), 1 );
			const box = new Box3( zero3, one3 );

			expect( a.intersectsBox( box ) ).toBe( true );
			expect( b.intersectsBox( box ) ).toBe( false );

		} );

		test( 'intersectsPlane', () => {

			const a = new Sphere( zero3.clone(), 1 );
			const b = new Plane( new Vector3( 0, 1, 0 ), 1 );
			const c = new Plane( new Vector3( 0, 1, 0 ), 1.25 );
			const d = new Plane( new Vector3( 0, - 1, 0 ), 1.25 );

			expect( a.intersectsPlane( b ) ).toBeTruthy();
			expect( ! a.intersectsPlane( c ) ).toBeTruthy();
			expect( ! a.intersectsPlane( d ) ).toBeTruthy();

		} );

		test( 'clampPoint', () => {

			const a = new Sphere( one3.clone(), 1 );
			const point = new Vector3();

			a.clampPoint( new Vector3( 1, 1, 3 ), point );
			expect( point.equals( new Vector3( 1, 1, 2 ) ) ).toBeTruthy();
			a.clampPoint( new Vector3( 1, 1, - 3 ), point );
			expect( point.equals( new Vector3( 1, 1, 0 ) ) ).toBeTruthy();

		} );

		test( 'getBoundingBox', () => {

			const a = new Sphere( one3.clone(), 1 );
			const aabb = new Box3();

			a.getBoundingBox( aabb );
			expect( aabb.equals( new Box3( zero3, two3 ) ) ).toBeTruthy();

			a.set( zero3, 0 );
			a.getBoundingBox( aabb );
			expect( aabb.equals( new Box3( zero3, zero3 ) ) ).toBeTruthy();

			// Empty sphere produces empty bounding box
			a.makeEmpty();
			a.getBoundingBox( aabb );
			expect( aabb.isEmpty() ).toBeTruthy();

		} );

		test( 'applyMatrix4', () => {

			const a = new Sphere( one3.clone(), 1 );
			const m = new Matrix4().makeTranslation( 1, - 2, 1 );
			const aabb1 = new Box3();
			const aabb2 = new Box3();

			a.clone().applyMatrix4( m ).getBoundingBox( aabb1 );
			a.getBoundingBox( aabb2 );

			expect( aabb1.equals( aabb2.applyMatrix4( m ) ) ).toBeTruthy();

		} );

		test( 'translate', () => {

			const a = new Sphere( one3.clone(), 1 );

			a.translate( one3.clone().negate() );
			expect( a.center.equals( zero3 ) ).toBeTruthy();

		} );

		test( 'expandByPoint', () => {

			const a = new Sphere( zero3.clone(), 1 );
			const p = new Vector3( 2, 0, 0 );

			expect( a.containsPoint( p ) === false ).toBeTruthy();

			a.expandByPoint( p );

			expect( a.containsPoint( p ) === true ).toBeTruthy();
			expect( a.center.equals( new Vector3( 0.5, 0, 0 ) ) ).toBeTruthy();
			expect( a.radius === 1.5 ).toBeTruthy();

		} );

		test( 'union', () => {

			const a = new Sphere( zero3.clone(), 1 );
			const b = new Sphere( new Vector3( 2, 0, 0 ), 1 );

			a.union( b );

			expect( a.center.equals( new Vector3( 1, 0, 0 ) ) ).toBeTruthy();
			expect( a.radius === 2 ).toBeTruthy();

			// d contains c (demonstrates why it is necessary to process two points in union)

			const c = new Sphere( new Vector3(), 1 );
			const d = new Sphere( new Vector3( 1, 0, 0 ), 4 );

			c.union( d );

			expect( c.center.equals( new Vector3( 1, 0, 0 ) ) ).toBeTruthy();
			expect( c.radius === 4 ).toBeTruthy();

			// edge case: both spheres have the same center point

			const e = new Sphere( new Vector3(), 1 );
			const f = new Sphere( new Vector3(), 4 );

			e.union( f );

			expect( e.center.equals( new Vector3( 0, 0, 0 ) ) ).toBeTruthy();
			expect( e.radius === 4 ).toBeTruthy();

		} );

		test( 'equals', () => {

			const a = new Sphere();
			const b = new Sphere( new Vector3( 1, 0, 0 ) );
			const c = new Sphere( new Vector3( 1, 0, 0 ), 1.0 );

			expect( a.equals( b ) ).toBe( false );
			expect( a.equals( c ) ).toBe( false );
			expect( b.equals( c ) ).toBe( false );

			a.copy( b );
			expect( a.equals( b ) ).toBe( true );

		} );

	} );

} );

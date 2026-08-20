import { describe, test, expect } from 'vitest';
import { BufferAttribute } from '@src/core/BufferAttribute.js';
import { Triangle } from '@src/math/Triangle.js';
import { Box3 } from '@src/math/Box3.js';
import { Plane } from '@src/math/Plane.js';
import { Vector3 } from '@src/math/Vector3.js';
import {
	zero3,
	one3,
	two3
} from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Triangle', () => {

		test( 'Instancing', () => {

			let a = new Triangle();
			expect( a.a.equals( zero3 ) ).toBeTruthy();
			expect( a.b.equals( zero3 ) ).toBeTruthy();
			expect( a.c.equals( zero3 ) ).toBeTruthy();

			a = new Triangle( one3.clone().negate(), one3.clone(), two3.clone() );
			expect( a.a.equals( one3.clone().negate() ) ).toBeTruthy();
			expect( a.b.equals( one3 ) ).toBeTruthy();
			expect( a.c.equals( two3 ) ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Triangle();

			a.set( one3.clone().negate(), one3, two3 );
			expect( a.a.equals( one3.clone().negate() ) ).toBeTruthy();
			expect( a.b.equals( one3 ) ).toBeTruthy();
			expect( a.c.equals( two3 ) ).toBeTruthy();

		} );

		test( 'setFromPointsAndIndices', () => {

			const a = new Triangle();

			const points = [ one3, one3.clone().negate(), two3 ];
			a.setFromPointsAndIndices( points, 1, 0, 2 );
			expect( a.a.equals( one3.clone().negate() ) ).toBeTruthy();
			expect( a.b.equals( one3 ) ).toBeTruthy();
			expect( a.c.equals( two3 ) ).toBeTruthy();

		} );

		test( 'setFromAttributeAndIndices', () => {

			const a = new Triangle();
			const attribute = new BufferAttribute( new Float32Array( [ 1, 1, 1, - 1, - 1, - 1, 2, 2, 2 ] ), 3 );

			a.setFromAttributeAndIndices( attribute, 1, 0, 2 );
			expect( a.a.equals( one3.clone().negate() ) ).toBeTruthy();
			expect( a.b.equals( one3 ) ).toBeTruthy();
			expect( a.c.equals( two3 ) ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Triangle( one3.clone().negate(), one3.clone(), two3.clone() );
			const b = new Triangle().copy( a );
			expect( b.a.equals( one3.clone().negate() ) ).toBeTruthy();
			expect( b.b.equals( one3 ) ).toBeTruthy();
			expect( b.c.equals( two3 ) ).toBeTruthy();

			// ensure that it is a true copy
			a.a = one3;
			a.b = zero3;
			a.c = zero3;
			expect( b.a.equals( one3.clone().negate() ) ).toBeTruthy();
			expect( b.b.equals( one3 ) ).toBeTruthy();
			expect( b.c.equals( two3 ) ).toBeTruthy();

		} );

		test( 'getArea', () => {

			let a = new Triangle();

			expect( a.getArea() == 0 ).toBeTruthy();

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			expect( a.getArea() == 0.5 ).toBeTruthy();

			a = new Triangle( new Vector3( 2, 0, 0 ), new Vector3( 0, 0, 0 ), new Vector3( 0, 0, 2 ) );
			expect( a.getArea() == 2 ).toBeTruthy();

			// colinear triangle.
			a = new Triangle( new Vector3( 2, 0, 0 ), new Vector3( 0, 0, 0 ), new Vector3( 3, 0, 0 ) );
			expect( a.getArea() == 0 ).toBeTruthy();

		} );

		test( 'getMidpoint', () => {

			let a = new Triangle();
			const midpoint = new Vector3();

			expect( a.getMidpoint( midpoint ).equals( new Vector3( 0, 0, 0 ) ) ).toBeTruthy();

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			expect( a.getMidpoint( midpoint ).equals( new Vector3( 1 / 3, 1 / 3, 0 ) ) ).toBeTruthy();

			a = new Triangle( new Vector3( 2, 0, 0 ), new Vector3( 0, 0, 0 ), new Vector3( 0, 0, 2 ) );
			expect( a.getMidpoint( midpoint ).equals( new Vector3( 2 / 3, 0, 2 / 3 ) ) ).toBeTruthy();

		} );

		test( 'getNormal', () => {

			let a = new Triangle();
			const normal = new Vector3();

			expect( a.getNormal( normal ).equals( new Vector3( 0, 0, 0 ) ) ).toBeTruthy();

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			expect( a.getNormal( normal ).equals( new Vector3( 0, 0, 1 ) ) ).toBeTruthy();

			a = new Triangle( new Vector3( 2, 0, 0 ), new Vector3( 0, 0, 0 ), new Vector3( 0, 0, 2 ) );
			expect( a.getNormal( normal ).equals( new Vector3( 0, 1, 0 ) ) ).toBeTruthy();

		} );

		test( 'getPlane', () => {

			let a = new Triangle();
			const plane = new Plane();
			const normal = new Vector3();

			a.getPlane( plane );
			expect( isNaN( plane.distanceToPoint( a.a ) ) ).toBeFalsy();
			expect( isNaN( plane.distanceToPoint( a.b ) ) ).toBeFalsy();
			expect( isNaN( plane.distanceToPoint( a.c ) ) ).toBeFalsy();
			expect( plane.normal ).not.toEqual( {
				x: NaN,
				y: NaN,
				z: NaN
			} );

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			a.getPlane( plane );
			a.getNormal( normal );
			expect( plane.distanceToPoint( a.a ) == 0 ).toBeTruthy();
			expect( plane.distanceToPoint( a.b ) == 0 ).toBeTruthy();
			expect( plane.distanceToPoint( a.c ) == 0 ).toBeTruthy();
			expect( plane.normal.equals( normal ) ).toBeTruthy();

			a = new Triangle( new Vector3( 2, 0, 0 ), new Vector3( 0, 0, 0 ), new Vector3( 0, 0, 2 ) );
			a.getPlane( plane );
			a.getNormal( normal );
			expect( plane.distanceToPoint( a.a ) == 0 ).toBeTruthy();
			expect( plane.distanceToPoint( a.b ) == 0 ).toBeTruthy();
			expect( plane.distanceToPoint( a.c ) == 0 ).toBeTruthy();
			expect( plane.normal.clone().normalize().equals( normal ) ).toBeTruthy();

		} );

		test( 'getBarycoord', () => {

			let a = new Triangle();

			const barycoord = new Vector3();
			const midpoint = new Vector3();

			expect( a.getBarycoord( a.a, barycoord ) === null ).toBeTruthy();
			expect( a.getBarycoord( a.b, barycoord ) === null ).toBeTruthy();
			expect( a.getBarycoord( a.c, barycoord ) === null ).toBeTruthy();

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			a.getMidpoint( midpoint );

			a.getBarycoord( a.a, barycoord );
			expect( barycoord.equals( new Vector3( 1, 0, 0 ) ) ).toBeTruthy();
			a.getBarycoord( a.b, barycoord );
			expect( barycoord.equals( new Vector3( 0, 1, 0 ) ) ).toBeTruthy();
			a.getBarycoord( a.c, barycoord );
			expect( barycoord.equals( new Vector3( 0, 0, 1 ) ) ).toBeTruthy();
			a.getBarycoord( midpoint, barycoord );
			expect( barycoord.distanceTo( new Vector3( 1 / 3, 1 / 3, 1 / 3 ) ) < 0.0001 ).toBeTruthy();

			a = new Triangle( new Vector3( 2, 0, 0 ), new Vector3( 0, 0, 0 ), new Vector3( 0, 0, 2 ) );
			a.getMidpoint( midpoint );

			a.getBarycoord( a.a, barycoord );
			expect( barycoord.equals( new Vector3( 1, 0, 0 ) ) ).toBeTruthy();
			a.getBarycoord( a.b, barycoord );
			expect( barycoord.equals( new Vector3( 0, 1, 0 ) ) ).toBeTruthy();
			a.getBarycoord( a.c, barycoord );
			expect( barycoord.equals( new Vector3( 0, 0, 1 ) ) ).toBeTruthy();
			a.getBarycoord( midpoint, barycoord );
			expect( barycoord.distanceTo( new Vector3( 1 / 3, 1 / 3, 1 / 3 ) ) < 0.0001 ).toBeTruthy();

		} );

		test( 'containsPoint', () => {

			let a = new Triangle();
			const midpoint = new Vector3();

			expect( ! a.containsPoint( a.a ) ).toBeTruthy();
			expect( ! a.containsPoint( a.b ) ).toBeTruthy();
			expect( ! a.containsPoint( a.c ) ).toBeTruthy();

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			a.getMidpoint( midpoint );
			expect( a.containsPoint( a.a ) ).toBeTruthy();
			expect( a.containsPoint( a.b ) ).toBeTruthy();
			expect( a.containsPoint( a.c ) ).toBeTruthy();
			expect( a.containsPoint( midpoint ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( - 1, - 1, - 1 ) ) ).toBeTruthy();

			a = new Triangle( new Vector3( 2, 0, 0 ), new Vector3( 0, 0, 0 ), new Vector3( 0, 0, 2 ) );
			a.getMidpoint( midpoint );
			expect( a.containsPoint( a.a ) ).toBeTruthy();
			expect( a.containsPoint( a.b ) ).toBeTruthy();
			expect( a.containsPoint( a.c ) ).toBeTruthy();
			expect( a.containsPoint( midpoint ) ).toBeTruthy();
			expect( ! a.containsPoint( new Vector3( - 1, - 1, - 1 ) ) ).toBeTruthy();

		} );

		test( 'intersectsBox', () => {

			const a = new Box3( one3.clone(), two3.clone() );
			const b = new Triangle( new Vector3( 1.5, 1.5, 2.5 ), new Vector3( 2.5, 1.5, 1.5 ), new Vector3( 1.5, 2.5, 1.5 ) );
			const c = new Triangle( new Vector3( 1.5, 1.5, 3.5 ), new Vector3( 3.5, 1.5, 1.5 ), new Vector3( 1.5, 1.5, 1.5 ) );
			const d = new Triangle( new Vector3( 1.5, 1.75, 3 ), new Vector3( 3, 1.75, 1.5 ), new Vector3( 1.5, 2.5, 1.5 ) );
			const e = new Triangle( new Vector3( 1.5, 1.8, 3 ), new Vector3( 3, 1.8, 1.5 ), new Vector3( 1.5, 2.5, 1.5 ) );
			const f = new Triangle( new Vector3( 1.5, 2.5, 3 ), new Vector3( 3, 2.5, 1.5 ), new Vector3( 1.5, 2.5, 1.5 ) );

			expect( b.intersectsBox( a ) ).toBeTruthy();
			expect( c.intersectsBox( a ) ).toBeTruthy();
			expect( d.intersectsBox( a ) ).toBeTruthy();
			expect( ! e.intersectsBox( a ) ).toBeTruthy();
			expect( ! f.intersectsBox( a ) ).toBeTruthy();

		} );

		test( 'closestPointToPoint', () => {

			const a = new Triangle( new Vector3( - 1, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			const point = new Vector3();

			// point lies inside the triangle
			a.closestPointToPoint( new Vector3( 0, 0.5, 0 ), point );
			expect( point.equals( new Vector3( 0, 0.5, 0 ) ) ).toBeTruthy();

			// point lies on a vertex
			a.closestPointToPoint( a.a, point );
			expect( point.equals( a.a ) ).toBeTruthy();

			a.closestPointToPoint( a.b, point );
			expect( point.equals( a.b ) ).toBeTruthy();

			a.closestPointToPoint( a.c, point );
			expect( point.equals( a.c ) ).toBeTruthy();

			// point lies on an edge
			a.closestPointToPoint( zero3.clone(), point );
			expect( point.equals( zero3.clone() ) ).toBeTruthy();

			// point lies outside the triangle
			a.closestPointToPoint( new Vector3( - 2, 0, 0 ), point );
			expect( point.equals( new Vector3( - 1, 0, 0 ) ) ).toBeTruthy();

			a.closestPointToPoint( new Vector3( 2, 0, 0 ), point );
			expect( point.equals( new Vector3( 1, 0, 0 ) ) ).toBeTruthy();

			a.closestPointToPoint( new Vector3( 0, 2, 0 ), point );
			expect( point.equals( new Vector3( 0, 1, 0 ) ) ).toBeTruthy();

			a.closestPointToPoint( new Vector3( 0, - 2, 0 ), point );
			expect( point.equals( new Vector3( 0, 0, 0 ) ) ).toBeTruthy();

		} );

		test( 'isFrontFacing', () => {

			let a = new Triangle();
			let dir = new Vector3();
			expect( ! a.isFrontFacing( dir ) ).toBeTruthy();

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ) );
			dir = new Vector3( 0, 0, - 1 );
			expect( a.isFrontFacing( dir ) ).toBeTruthy();

			a = new Triangle( new Vector3( 0, 0, 0 ), new Vector3( 0, 1, 0 ), new Vector3( 1, 0, 0 ) );
			expect( ! a.isFrontFacing( dir ) ).toBeTruthy();

		} );

		test( 'equals', () => {

			const a = new Triangle(
				new Vector3( 1, 0, 0 ),
				new Vector3( 0, 1, 0 ),
				new Vector3( 0, 0, 1 )
			);
			const b = new Triangle(
				new Vector3( 0, 0, 1 ),
				new Vector3( 0, 1, 0 ),
				new Vector3( 1, 0, 0 )
			);
			const c = new Triangle(
				new Vector3( - 1, 0, 0 ),
				new Vector3( 0, 1, 0 ),
				new Vector3( 0, 0, 1 )
			);

			expect( a.equals( a ) ).toBeTruthy();
			expect( a.equals( b ) ).toBeFalsy();
			expect( a.equals( c ) ).toBeFalsy();
			expect( b.equals( c ) ).toBeFalsy();

			a.copy( b );
			expect( a.equals( a ) ).toBeTruthy();

		} );

	} );

} );

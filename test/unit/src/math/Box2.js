import { describe, test, expect } from 'vitest';
import { Box2 } from '@src/math/Box2.js';
import { Vector2 } from '@src/math/Vector2.js';
import {
	negInf2,
	posInf2,
	negOne2,
	zero2,
	one2,
	two2
} from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Box2', () => {

		test( 'Instancing', () => {

			let a = new Box2();
			expect( a.min.equals( posInf2 ) ).toBeTruthy();
			expect( a.max.equals( negInf2 ) ).toBeTruthy();

			a = new Box2( zero2.clone(), zero2.clone() );
			expect( a.min.equals( zero2 ) ).toBeTruthy();
			expect( a.max.equals( zero2 ) ).toBeTruthy();

			a = new Box2( zero2.clone(), one2.clone() );
			expect( a.min.equals( zero2 ) ).toBeTruthy();
			expect( a.max.equals( one2 ) ).toBeTruthy();

		} );

		test( 'isBox2', () => {

			const a = new Box2();
			expect( a.isBox2 === true ).toBeTruthy();

			const b = new Object();
			expect( ! b.isBox2 ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Box2();

			a.set( zero2, one2 );
			expect( a.min.equals( zero2 ) ).toBeTruthy();
			expect( a.max.equals( one2 ) ).toBeTruthy();

		} );

		test( 'setFromPoints', () => {

			const a = new Box2();

			a.setFromPoints( [ zero2, one2, two2 ] );
			expect( a.min.equals( zero2 ) ).toBeTruthy();
			expect( a.max.equals( two2 ) ).toBeTruthy();

			a.setFromPoints( [ one2 ] );
			expect( a.min.equals( one2 ) ).toBeTruthy();
			expect( a.max.equals( one2 ) ).toBeTruthy();

			a.setFromPoints( [] );
			expect( a.isEmpty() ).toBeTruthy();

		} );

		test( 'setFromCenterAndSize', () => {

			const a = new Box2();

			a.setFromCenterAndSize( zero2, two2 );
			expect( a.min.equals( negOne2 ) ).toBeTruthy();
			expect( a.max.equals( one2 ) ).toBeTruthy();

			a.setFromCenterAndSize( one2, two2 );
			expect( a.min.equals( zero2 ) ).toBeTruthy();
			expect( a.max.equals( two2 ) ).toBeTruthy();

			a.setFromCenterAndSize( zero2, zero2 );
			expect( a.min.equals( zero2 ) ).toBeTruthy();
			expect( a.max.equals( zero2 ) ).toBeTruthy();

		} );

		test( 'clone', () => {

			let a = new Box2( zero2, zero2 );

			let b = a.clone();
			expect( b.min.equals( zero2 ) ).toBeTruthy();
			expect( b.max.equals( zero2 ) ).toBeTruthy();

			a = new Box2();
			b = a.clone();
			expect( b.min.equals( posInf2 ) ).toBeTruthy();
			expect( b.max.equals( negInf2 ) ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Box2( zero2.clone(), one2.clone() );
			const b = new Box2().copy( a );
			expect( b.min.equals( zero2 ) ).toBeTruthy();
			expect( b.max.equals( one2 ) ).toBeTruthy();

			// ensure that it is a true copy
			a.min = zero2;
			a.max = one2;
			expect( b.min.equals( zero2 ) ).toBeTruthy();
			expect( b.max.equals( one2 ) ).toBeTruthy();

		} );

		test( 'empty/makeEmpty', () => {

			let a = new Box2();

			expect( a.isEmpty() ).toBeTruthy();

			a = new Box2( zero2.clone(), one2.clone() );
			expect( ! a.isEmpty() ).toBeTruthy();

			a.makeEmpty();
			expect( a.isEmpty() ).toBeTruthy();

		} );

		test( 'isEmpty', () => {

			let a = new Box2( zero2.clone(), zero2.clone() );
			expect( ! a.isEmpty() ).toBeTruthy();

			a = new Box2( zero2.clone(), one2.clone() );
			expect( ! a.isEmpty() ).toBeTruthy();

			a = new Box2( two2.clone(), one2.clone() );
			expect( a.isEmpty() ).toBeTruthy();

			a = new Box2( posInf2.clone(), negInf2.clone() );
			expect( a.isEmpty() ).toBeTruthy();

		} );

		test( 'getCenter', () => {

			let a = new Box2( zero2.clone(), zero2.clone() );
			const center = new Vector2();
			expect( a.getCenter( center ).equals( zero2 ) ).toBeTruthy();

			a = new Box2( zero2, one2 );
			const midpoint = one2.clone().multiplyScalar( 0.5 );
			expect( a.getCenter( center ).equals( midpoint ) ).toBeTruthy();

		} );

		test( 'getSize', () => {

			let a = new Box2( zero2.clone(), zero2.clone() );
			const size = new Vector2();

			expect( a.getSize( size ).equals( zero2 ) ).toBeTruthy();

			a = new Box2( zero2.clone(), one2.clone() );
			expect( a.getSize( size ).equals( one2 ) ).toBeTruthy();

		} );

		test( 'expandByPoint', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const size = new Vector2();
			const center = new Vector2();

			a.expandByPoint( zero2 );
			expect( a.getSize( size ).equals( zero2 ) ).toBeTruthy();

			a.expandByPoint( one2 );
			expect( a.getSize( size ).equals( one2 ) ).toBeTruthy();

			a.expandByPoint( one2.clone().negate() );
			expect( a.getSize( size ).equals( one2.clone().multiplyScalar( 2 ) ) ).toBeTruthy();
			expect( a.getCenter( center ).equals( zero2 ) ).toBeTruthy();

		} );

		test( 'expandByVector', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const size = new Vector2();
			const center = new Vector2();

			a.expandByVector( zero2 );
			expect( a.getSize( size ).equals( zero2 ) ).toBeTruthy();

			a.expandByVector( one2 );
			expect( a.getSize( size ).equals( one2.clone().multiplyScalar( 2 ) ) ).toBeTruthy();
			expect( a.getCenter( center ).equals( zero2 ) ).toBeTruthy();

		} );

		test( 'expandByScalar', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const size = new Vector2();
			const center = new Vector2();

			a.expandByScalar( 0 );
			expect( a.getSize( size ).equals( zero2 ) ).toBeTruthy();

			a.expandByScalar( 1 );
			expect( a.getSize( size ).equals( one2.clone().multiplyScalar( 2 ) ) ).toBeTruthy();
			expect( a.getCenter( center ).equals( zero2 ) ).toBeTruthy();

		} );

		test( 'containsPoint', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );

			expect( a.containsPoint( zero2 ) ).toBeTruthy();
			expect( ! a.containsPoint( one2 ) ).toBeTruthy();

			a.expandByScalar( 1 );
			expect( a.containsPoint( zero2 ) ).toBeTruthy();
			expect( a.containsPoint( one2 ) ).toBeTruthy();
			expect( a.containsPoint( one2.clone().negate() ) ).toBeTruthy();

		} );

		test( 'containsBox', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const b = new Box2( zero2.clone(), one2.clone() );
			const c = new Box2( one2.clone().negate(), one2.clone() );

			expect( a.containsBox( a ) ).toBeTruthy();
			expect( ! a.containsBox( b ) ).toBeTruthy();
			expect( ! a.containsBox( c ) ).toBeTruthy();

			expect( b.containsBox( a ) ).toBeTruthy();
			expect( c.containsBox( a ) ).toBeTruthy();
			expect( ! b.containsBox( c ) ).toBeTruthy();

		} );

		test( 'getParameter', () => {

			const a = new Box2( zero2.clone(), one2.clone() );
			const b = new Box2( one2.clone().negate(), one2.clone() );

			const parameter = new Vector2();

			a.getParameter( zero2, parameter );
			expect( parameter.equals( zero2 ) ).toBeTruthy();
			a.getParameter( one2, parameter );
			expect( parameter.equals( one2 ) ).toBeTruthy();

			b.getParameter( one2.clone().negate(), parameter );
			expect( parameter.equals( zero2 ) ).toBeTruthy();
			b.getParameter( zero2, parameter );
			expect( parameter.equals( new Vector2( 0.5, 0.5 ) ) ).toBeTruthy();
			b.getParameter( one2, parameter );
			expect( parameter.equals( one2 ) ).toBeTruthy();

		} );

		test( 'intersectsBox', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const b = new Box2( zero2.clone(), one2.clone() );
			const c = new Box2( one2.clone().negate(), one2.clone() );

			expect( a.intersectsBox( a ) ).toBeTruthy();
			expect( a.intersectsBox( b ) ).toBeTruthy();
			expect( a.intersectsBox( c ) ).toBeTruthy();

			expect( b.intersectsBox( a ) ).toBeTruthy();
			expect( c.intersectsBox( a ) ).toBeTruthy();
			expect( b.intersectsBox( c ) ).toBeTruthy();

			b.translate( two2 );
			expect( ! a.intersectsBox( b ) ).toBeTruthy();
			expect( ! b.intersectsBox( a ) ).toBeTruthy();
			expect( ! b.intersectsBox( c ) ).toBeTruthy();

		} );

		test( 'clampPoint', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const b = new Box2( one2.clone().negate(), one2.clone() );

			const point = new Vector2();

			a.clampPoint( zero2, point );
			expect( point.equals( new Vector2( 0, 0 ) ) ).toBeTruthy();
			a.clampPoint( one2, point );
			expect( point.equals( new Vector2( 0, 0 ) ) ).toBeTruthy();
			a.clampPoint( one2.clone().negate(), point );
			expect( point.equals( new Vector2( 0, 0 ) ) ).toBeTruthy();

			b.clampPoint( two2, point );
			expect( point.equals( new Vector2( 1, 1 ) ) ).toBeTruthy();
			b.clampPoint( one2, point );
			expect( point.equals( new Vector2( 1, 1 ) ) ).toBeTruthy();
			b.clampPoint( zero2, point );
			expect( point.equals( new Vector2( 0, 0 ) ) ).toBeTruthy();
			b.clampPoint( one2.clone().negate(), point );
			expect( point.equals( new Vector2( - 1, - 1 ) ) ).toBeTruthy();
			b.clampPoint( two2.clone().negate(), point );
			expect( point.equals( new Vector2( - 1, - 1 ) ) ).toBeTruthy();

		} );

		test( 'distanceToPoint', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const b = new Box2( one2.clone().negate(), one2.clone() );

			expect( a.distanceToPoint( new Vector2( 0, 0 ) ) == 0 ).toBeTruthy();
			expect( a.distanceToPoint( new Vector2( 1, 1 ) ) == Math.sqrt( 2 ) ).toBeTruthy();
			expect( a.distanceToPoint( new Vector2( - 1, - 1 ) ) == Math.sqrt( 2 ) ).toBeTruthy();

			expect( b.distanceToPoint( new Vector2( 2, 2 ) ) == Math.sqrt( 2 ) ).toBeTruthy();
			expect( b.distanceToPoint( new Vector2( 1, 1 ) ) == 0 ).toBeTruthy();
			expect( b.distanceToPoint( new Vector2( 0, 0 ) ) == 0 ).toBeTruthy();
			expect( b.distanceToPoint( new Vector2( - 1, - 1 ) ) == 0 ).toBeTruthy();
			expect( b.distanceToPoint( new Vector2( - 2, - 2 ) ) == Math.sqrt( 2 ) ).toBeTruthy();

		} );

		test( 'intersect', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const b = new Box2( zero2.clone(), one2.clone() );
			const c = new Box2( one2.clone().negate(), one2.clone() );

			expect( a.clone().intersect( a ).equals( a ) ).toBeTruthy();
			expect( a.clone().intersect( b ).equals( a ) ).toBeTruthy();
			expect( b.clone().intersect( b ).equals( b ) ).toBeTruthy();
			expect( a.clone().intersect( c ).equals( a ) ).toBeTruthy();
			expect( b.clone().intersect( c ).equals( b ) ).toBeTruthy();
			expect( c.clone().intersect( c ).equals( c ) ).toBeTruthy();

			const d = new Box2( one2.clone().negate(), zero2.clone() );
			const e = new Box2( one2.clone(), two2.clone() ).intersect( d );

			expect( e.min.equals( posInf2 ) && e.max.equals( negInf2 ) ).toBeTruthy();

		} );

		test( 'union', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const b = new Box2( zero2.clone(), one2.clone() );
			const c = new Box2( one2.clone().negate(), one2.clone() );

			expect( a.clone().union( a ).equals( a ) ).toBeTruthy();
			expect( a.clone().union( b ).equals( b ) ).toBeTruthy();
			expect( a.clone().union( c ).equals( c ) ).toBeTruthy();
			expect( b.clone().union( c ).equals( c ) ).toBeTruthy();

		} );

		test( 'translate', () => {

			const a = new Box2( zero2.clone(), zero2.clone() );
			const b = new Box2( zero2.clone(), one2.clone() );
			const c = new Box2( one2.clone().negate(), zero2.clone() );

			expect( a.clone().translate( one2 ).equals( new Box2( one2, one2 ) ) ).toBeTruthy();
			expect( a.clone().translate( one2 ).translate( one2.clone().negate() ).equals( a ) ).toBeTruthy();
			expect( c.clone().translate( one2 ).equals( b ) ).toBeTruthy();
			expect( b.clone().translate( one2.clone().negate() ).equals( c ) ).toBeTruthy();

		} );

		test( 'equals', () => {

			let a = new Box2();
			let b = new Box2();
			expect( b.equals( a ) ).toBeTruthy();
			expect( a.equals( b ) ).toBeTruthy();

			a = new Box2( one2, two2 );
			b = new Box2( one2, two2 );
			expect( b.equals( a ) ).toBeTruthy();
			expect( a.equals( b ) ).toBeTruthy();

			a = new Box2( one2, two2 );
			b = a.clone();
			expect( b.equals( a ) ).toBeTruthy();
			expect( a.equals( b ) ).toBeTruthy();

			a = new Box2( one2, two2 );
			b = new Box2( one2, one2 );
			expect( ! b.equals( a ) ).toBeTruthy();
			expect( ! a.equals( b ) ).toBeTruthy();

			a = new Box2();
			b = new Box2( one2, one2 );
			expect( ! b.equals( a ) ).toBeTruthy();
			expect( ! a.equals( b ) ).toBeTruthy();

			a = new Box2( one2, two2 );
			b = new Box2( one2, one2 );
			expect( ! b.equals( a ) ).toBeTruthy();
			expect( ! a.equals( b ) ).toBeTruthy();

		} );

	} );

} );

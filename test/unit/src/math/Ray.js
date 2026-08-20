import { describe, test, expect } from 'vitest';
import { Ray } from '@src/math/Ray.js';
import { Box3 } from '@src/math/Box3.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Sphere } from '@src/math/Sphere.js';
import { Plane } from '@src/math/Plane.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import {
	zero3,
	one3,
	two3,
	eps,
	posInf3
} from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Ray', () => {

		test( 'Instancing', () => {

			let a = new Ray();
			expect( a.origin.equals( zero3 ) ).toBeTruthy();
			expect( a.direction.equals( new Vector3( 0, 0, - 1 ) ) ).toBeTruthy();

			a = new Ray( two3.clone(), one3.clone() );
			expect( a.origin.equals( two3 ) ).toBeTruthy();
			expect( a.direction.equals( one3 ) ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Ray();

			a.set( one3, one3 );
			expect( a.origin.equals( one3 ) ).toBeTruthy();
			expect( a.direction.equals( one3 ) ).toBeTruthy();

		} );

		test( 'recast/clone', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );

			expect( a.recast( 0 ).equals( a ) ).toBeTruthy();

			const b = a.clone();
			expect( b.recast( - 1 ).equals( new Ray( new Vector3( 1, 1, 0 ), new Vector3( 0, 0, 1 ) ) ) ).toBeTruthy();

			const c = a.clone();
			expect( c.recast( 1 ).equals( new Ray( new Vector3( 1, 1, 2 ), new Vector3( 0, 0, 1 ) ) ) ).toBeTruthy();

			const d = a.clone();
			const e = d.clone().recast( 1 );
			expect( d.equals( a ) ).toBeTruthy();
			expect( ! e.equals( d ) ).toBeTruthy();
			expect( e.equals( c ) ).toBeTruthy();

		} );

		test( 'copy/equals', () => {

			const a = new Ray( zero3.clone(), one3.clone() );
			const b = new Ray().copy( a );
			expect( b.origin.equals( zero3 ) ).toBeTruthy();
			expect( b.direction.equals( one3 ) ).toBeTruthy();

			// ensure that it is a true copy
			a.origin = zero3;
			a.direction = one3;
			expect( b.origin.equals( zero3 ) ).toBeTruthy();
			expect( b.direction.equals( one3 ) ).toBeTruthy();

		} );

		test( 'at', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );
			const point = new Vector3();

			a.at( 0, point );
			expect( point.equals( one3 ) ).toBeTruthy();
			a.at( - 1, point );
			expect( point.equals( new Vector3( 1, 1, 0 ) ) ).toBeTruthy();
			a.at( 1, point );
			expect( point.equals( new Vector3( 1, 1, 2 ) ) ).toBeTruthy();

		} );

		test( 'lookAt', () => {

			const a = new Ray( two3.clone(), one3.clone() );
			const target = one3.clone();
			const expected = target.sub( two3 ).normalize();

			a.lookAt( target );
			expect( a.direction.equals( expected ) ).toBeTruthy();

		} );

		test( 'closestPointToPoint', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );
			const point = new Vector3();

			// behind the ray
			a.closestPointToPoint( zero3, point );
			expect( point.equals( one3 ) ).toBeTruthy();

			// front of the ray
			a.closestPointToPoint( new Vector3( 0, 0, 50 ), point );
			expect( point.equals( new Vector3( 1, 1, 50 ) ) ).toBeTruthy();

			// exactly on the ray
			a.closestPointToPoint( one3, point );
			expect( point.equals( one3 ) ).toBeTruthy();

		} );

		test( 'distanceToPoint', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );

			// behind the ray
			const b = a.distanceToPoint( zero3 );
			expect( b === Math.sqrt( 3 ) ).toBeTruthy();

			// front of the ray
			const c = a.distanceToPoint( new Vector3( 0, 0, 50 ) );
			expect( c === Math.sqrt( 2 ) ).toBeTruthy();

			// exactly on the ray
			const d = a.distanceToPoint( one3 );
			expect( d === 0 ).toBeTruthy();

		} );

		test( 'distanceSqToPoint', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );

			// behind the ray
			const b = a.distanceSqToPoint( zero3 );
			expect( b === 3 ).toBeTruthy();

			// front of the ray
			const c = a.distanceSqToPoint( new Vector3( 0, 0, 50 ) );
			expect( c === 2 ).toBeTruthy();

			// exactly on the ray
			const d = a.distanceSqToPoint( one3 );
			expect( d === 0 ).toBeTruthy();

		} );

		test( 'distanceSqToSegment', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );
			const ptOnLine = new Vector3();
			const ptOnSegment = new Vector3();

			//segment in front of the ray
			let v0 = new Vector3( 3, 5, 50 );
			let v1 = new Vector3( 50, 50, 50 ); // just a far away point
			let distSqr = a.distanceSqToSegment( v0, v1, ptOnLine, ptOnSegment );

			expect( ptOnSegment.distanceTo( v0 ) < 0.0001 ).toBeTruthy();
			expect( ptOnLine.distanceTo( new Vector3( 1, 1, 50 ) ) < 0.0001 ).toBeTruthy();
			// ((3-1) * (3-1) + (5-1) * (5-1) = 4 + 16 = 20
			expect( Math.abs( distSqr - 20 ) < 0.0001 ).toBeTruthy();

			//segment behind the ray
			v0 = new Vector3( - 50, - 50, - 50 ); // just a far away point
			v1 = new Vector3( - 3, - 5, - 4 );
			distSqr = a.distanceSqToSegment( v0, v1, ptOnLine, ptOnSegment );

			expect( ptOnSegment.distanceTo( v1 ) < 0.0001 ).toBeTruthy();
			expect( ptOnLine.distanceTo( one3 ) < 0.0001 ).toBeTruthy();
			// ((-3-1) * (-3-1) + (-5-1) * (-5-1) + (-4-1) + (-4-1) = 16 + 36 + 25 = 77
			expect( Math.abs( distSqr - 77 ) < 0.0001 ).toBeTruthy();

			//exact intersection between the ray and the segment
			v0 = new Vector3( - 50, - 50, - 50 );
			v1 = new Vector3( 50, 50, 50 );
			distSqr = a.distanceSqToSegment( v0, v1, ptOnLine, ptOnSegment );

			expect( ptOnSegment.distanceTo( one3 ) < 0.0001 ).toBeTruthy();
			expect( ptOnLine.distanceTo( one3 ) < 0.0001 ).toBeTruthy();
			expect( distSqr < 0.0001 ).toBeTruthy();

		} );

		test( 'intersectSphere', () => {

			const TOL = 0.0001;
			const point = new Vector3();

			// ray a0 origin located at ( 0, 0, 0 ) and points outward in negative-z direction
			const a0 = new Ray( zero3.clone(), new Vector3( 0, 0, - 1 ) );
			// ray a1 origin located at ( 1, 1, 1 ) and points left in negative-x direction
			const a1 = new Ray( one3.clone(), new Vector3( - 1, 0, 0 ) );

			// sphere (radius of 2) located behind ray a0, should result in null
			let b = new Sphere( new Vector3( 0, 0, 3 ), 2 );
			a0.intersectSphere( b, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// sphere (radius of 2) located in front of, but too far right of ray a0, should result in null
			b = new Sphere( new Vector3( 3, 0, - 1 ), 2 );
			a0.intersectSphere( b, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// sphere (radius of 2) located below ray a1, should result in null
			b = new Sphere( new Vector3( 1, - 2, 1 ), 2 );
			a1.intersectSphere( b, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// sphere (radius of 1) located to the left of ray a1, should result in intersection at 0, 1, 1
			b = new Sphere( new Vector3( - 1, 1, 1 ), 1 );
			a1.intersectSphere( b, point );
			expect( point.distanceTo( new Vector3( 0, 1, 1 ) ) < TOL ).toBeTruthy();

			// sphere (radius of 1) located in front of ray a0, should result in intersection at 0, 0, -1
			b = new Sphere( new Vector3( 0, 0, - 2 ), 1 );
			a0.intersectSphere( b, point );
			expect( point.distanceTo( new Vector3( 0, 0, - 1 ) ) < TOL ).toBeTruthy();

			// sphere (radius of 2) located in front & right of ray a0, should result in intersection at 0, 0, -1, or left-most edge of sphere
			b = new Sphere( new Vector3( 2, 0, - 1 ), 2 );
			a0.intersectSphere( b, point );
			expect( point.distanceTo( new Vector3( 0, 0, - 1 ) ) < TOL ).toBeTruthy();

			// same situation as above, but move the sphere a fraction more to the right, and ray a0 should now just miss
			b = new Sphere( new Vector3( 2.01, 0, - 1 ), 2 );
			a0.intersectSphere( b, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// following tests are for situations where the ray origin is inside the sphere

			// sphere (radius of 1) center located at ray a0 origin / sphere surrounds the ray origin, so the first intersect point 0, 0, 1,
			// is behind ray a0.  Therefore, second exit point on back of sphere will be returned: 0, 0, -1
			// thus keeping the intersection point always in front of the ray.
			b = new Sphere( zero3.clone(), 1 );
			a0.intersectSphere( b, point );
			expect( point.distanceTo( new Vector3( 0, 0, - 1 ) ) < TOL ).toBeTruthy();

			// sphere (radius of 4) center located behind ray a0 origin / sphere surrounds the ray origin, so the first intersect point 0, 0, 5,
			// is behind ray a0.  Therefore, second exit point on back of sphere will be returned: 0, 0, -3
			// thus keeping the intersection point always in front of the ray.
			b = new Sphere( new Vector3( 0, 0, 1 ), 4 );
			a0.intersectSphere( b, point );
			expect( point.distanceTo( new Vector3( 0, 0, - 3 ) ) < TOL ).toBeTruthy();

			// sphere (radius of 4) center located in front of ray a0 origin / sphere surrounds the ray origin, so the first intersect point 0, 0, 3,
			// is behind ray a0.  Therefore, second exit point on back of sphere will be returned: 0, 0, -5
			// thus keeping the intersection point always in front of the ray.
			b = new Sphere( new Vector3( 0, 0, - 1 ), 4 );
			a0.intersectSphere( b, point );
			expect( point.distanceTo( new Vector3( 0, 0, - 5 ) ) < TOL ).toBeTruthy();

			// empty sphere ( negative radius ) can never be intersected, so the
			// target must be left untouched and the return value must be null,
			// consistent with intersectsSphere()
			b = new Sphere( new Vector3( 0, 0, - 1 ), - 1 );
			expect( a0.intersectSphere( b, point.copy( posInf3 ) ) ).toBe( null );
			expect( point.equals( posInf3 ) ).toBeTruthy();

		} );

		test( 'intersectsSphere', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );
			const b = new Sphere( zero3, 0.5 );
			const c = new Sphere( zero3, 1.5 );
			const d = new Sphere( one3, 0.1 );
			const e = new Sphere( two3, 0.1 );
			const f = new Sphere( two3, 1 );

			expect( ! a.intersectsSphere( b ) ).toBeTruthy();
			expect( ! a.intersectsSphere( c ) ).toBeTruthy();
			expect( a.intersectsSphere( d ) ).toBeTruthy();
			expect( ! a.intersectsSphere( e ) ).toBeTruthy();
			expect( ! a.intersectsSphere( f ) ).toBeTruthy();

		} );

		test( 'intersectPlane', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );
			const point = new Vector3();

			// parallel plane behind
			const b = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 0, 0, 1 ), new Vector3( 1, 1, - 1 ) );
			a.intersectPlane( b, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// parallel plane coincident with origin
			const c = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 0, 0, 1 ), new Vector3( 1, 1, 0 ) );
			a.intersectPlane( c, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// parallel plane in front
			const d = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 0, 0, 1 ), new Vector3( 1, 1, 1 ) );
			a.intersectPlane( d, point.copy( posInf3 ) );
			expect( point.equals( a.origin ) ).toBeTruthy();

			// perpendicular ray that overlaps exactly
			const e = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 1, 0, 0 ), one3 );
			a.intersectPlane( e, point.copy( posInf3 ) );
			expect( point.equals( a.origin ) ).toBeTruthy();

			// perpendicular ray that doesn't overlap
			const f = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 1, 0, 0 ), zero3 );
			a.intersectPlane( f, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

		} );

		test( 'intersectsPlane', () => {

			const a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );

			// parallel plane in front of the ray
			const b = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 0, 0, 1 ), one3.clone().sub( new Vector3( 0, 0, - 1 ) ) );
			expect( a.intersectsPlane( b ) ).toBeTruthy();

			// parallel plane coincident with origin
			const c = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 0, 0, 1 ), one3.clone().sub( new Vector3( 0, 0, 0 ) ) );
			expect( a.intersectsPlane( c ) ).toBeTruthy();

			// parallel plane behind the ray
			const d = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 0, 0, 1 ), one3.clone().sub( new Vector3( 0, 0, 1 ) ) );
			expect( ! a.intersectsPlane( d ) ).toBeTruthy();

			// perpendicular ray that overlaps exactly
			const e = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 1, 0, 0 ), one3 );
			expect( a.intersectsPlane( e ) ).toBeTruthy();

			// perpendicular ray that doesn't overlap
			const f = new Plane().setFromNormalAndCoplanarPoint( new Vector3( 1, 0, 0 ), zero3 );
			expect( ! a.intersectsPlane( f ) ).toBeTruthy();

		} );

		test( 'intersectBox', () => {

			const TOL = 0.0001;

			const box = new Box3( new Vector3( - 1, - 1, - 1 ), new Vector3( 1, 1, 1 ) );
			const point = new Vector3();

			const a = new Ray( new Vector3( - 2, 0, 0 ), new Vector3( 1, 0, 0 ) );
			//ray should intersect box at -1,0,0
			expect( a.intersectsBox( box ) === true ).toBeTruthy();
			a.intersectBox( box, point );
			expect( point.distanceTo( new Vector3( - 1, 0, 0 ) ) < TOL ).toBeTruthy();

			const b = new Ray( new Vector3( - 2, 0, 0 ), new Vector3( - 1, 0, 0 ) );
			//ray is point away from box, it should not intersect
			expect( b.intersectsBox( box ) === false ).toBeTruthy();
			b.intersectBox( box, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			const c = new Ray( new Vector3( 0, 0, 0 ), new Vector3( 1, 0, 0 ) );
			// ray is inside box, should return exit point
			expect( c.intersectsBox( box ) === true ).toBeTruthy();
			c.intersectBox( box, point );
			expect( point.distanceTo( new Vector3( 1, 0, 0 ) ) < TOL ).toBeTruthy();

			const d = new Ray( new Vector3( 0, 2, 1 ), new Vector3( 0, - 1, - 1 ).normalize() );
			//tilted ray should intersect box at 0,1,0
			expect( d.intersectsBox( box ) === true ).toBeTruthy();
			d.intersectBox( box, point );
			expect( point.distanceTo( new Vector3( 0, 1, 0 ) ) < TOL ).toBeTruthy();

			const e = new Ray( new Vector3( 1, - 2, 1 ), new Vector3( 0, 1, 0 ).normalize() );
			//handle case where ray is coplanar with one of the boxes side - box in front of ray
			expect( e.intersectsBox( box ) === true ).toBeTruthy();
			e.intersectBox( box, point );
			expect( point.distanceTo( new Vector3( 1, - 1, 1 ) ) < TOL ).toBeTruthy();

			const f = new Ray( new Vector3( 1, - 2, 0 ), new Vector3( 0, - 1, 0 ).normalize() );
			//handle case where ray is coplanar with one of the boxes side - box behind ray
			expect( f.intersectsBox( box ) === false ).toBeTruthy();
			f.intersectBox( box, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

		} );

		test( 'intersectTriangle', () => {

			const ray = new Ray();
			const a = new Vector3( 1, 1, 0 );
			const b = new Vector3( 0, 1, 1 );
			const c = new Vector3( 1, 0, 1 );
			const point = new Vector3();

			// DdN == 0
			ray.set( ray.origin, zero3.clone() );
			ray.intersectTriangle( a, b, c, false, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// DdN > 0, backfaceCulling = true
			ray.set( ray.origin, one3.clone() );
			ray.intersectTriangle( a, b, c, true, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// DdN > 0
			ray.set( ray.origin, one3.clone() );
			ray.intersectTriangle( a, b, c, false, point );
			expect( Math.abs( point.x - 2 / 3 ) <= eps ).toBeTruthy();
			expect( Math.abs( point.y - 2 / 3 ) <= eps ).toBeTruthy();
			expect( Math.abs( point.z - 2 / 3 ) <= eps ).toBeTruthy();

			// DdN > 0, DdQxE2 < 0
			b.multiplyScalar( - 1 );
			ray.intersectTriangle( a, b, c, false, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// DdN > 0, DdE1xQ < 0
			a.multiplyScalar( - 1 );
			ray.intersectTriangle( a, b, c, false, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// DdN > 0, DdQxE2 + DdE1xQ > DdN
			b.multiplyScalar( - 1 );
			ray.intersectTriangle( a, b, c, false, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

			// DdN < 0, QdN < 0
			a.multiplyScalar( - 1 );
			b.multiplyScalar( - 1 );
			ray.direction.multiplyScalar( - 1 );
			ray.intersectTriangle( a, b, c, false, point.copy( posInf3 ) );
			expect( point.equals( posInf3 ) ).toBeTruthy();

		} );

		test( 'intersectTriangle (watertight at shared edges)', () => {

			// Two triangles forming a quad and sharing the diagonal edge from
			// ( -2, -2, -2 ) to ( 2, -2, 2 ). A ray aimed exactly at the midpoint of
			// that shared edge must be detected: a non-watertight test can let the ray
			// slip through the seam between the triangles and miss both of them.

			const t1a = new Vector3( - 2, - 2, 2 );
			const t1b = new Vector3( - 2, - 2, - 2 );
			const t1c = new Vector3( 2, - 2, 2 );

			const t2a = new Vector3( - 2, - 2, - 2 );
			const t2b = new Vector3( 2, - 2, - 2 );
			const t2c = new Vector3( 2, - 2, 2 );

			const seam = new Vector3( 0, - 2, 0 ); // midpoint of the shared edge
			const origin = new Vector3( - 4, - 9, 0.4 );
			const direction = new Vector3().subVectors( seam, origin ).normalize();
			const ray = new Ray( origin, direction );

			const p1 = new Vector3();
			const p2 = new Vector3();
			const hit1 = ray.intersectTriangle( t1a, t1b, t1c, false, p1 );
			const hit2 = ray.intersectTriangle( t2a, t2b, t2c, false, p2 );

			expect( hit1 !== null || hit2 !== null ).toBeTruthy();

			const hit = hit1 !== null ? p1 : p2;
			expect( hit.distanceTo( seam ) <= eps ).toBeTruthy();

		} );

		test( 'applyMatrix4', () => {

			let a = new Ray( one3.clone(), new Vector3( 0, 0, 1 ) );
			const m = new Matrix4();

			expect( a.clone().applyMatrix4( m ).equals( a ) ).toBeTruthy();

			a = new Ray( zero3.clone(), new Vector3( 0, 0, 1 ) );
			m.makeRotationZ( Math.PI );
			expect( a.clone().applyMatrix4( m ).equals( a ) ).toBeTruthy();

			m.makeRotationX( Math.PI );
			const b = a.clone();
			b.direction.negate();
			let a2 = a.clone().applyMatrix4( m );
			expect( a2.origin.distanceTo( b.origin ) < 0.0001 ).toBeTruthy();
			expect( a2.direction.distanceTo( b.direction ) < 0.0001 ).toBeTruthy();

			a.origin = new Vector3( 0, 0, 1 );
			b.origin = new Vector3( 0, 0, - 1 );
			a2 = a.clone().applyMatrix4( m );
			expect( a2.origin.distanceTo( b.origin ) < 0.0001 ).toBeTruthy();
			expect( a2.direction.distanceTo( b.direction ) < 0.0001 ).toBeTruthy();

		} );

	} );

} );

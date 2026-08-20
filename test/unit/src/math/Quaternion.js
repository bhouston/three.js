import { describe, test, expect } from 'vitest';
import { BufferAttribute } from '@src/core/BufferAttribute.js';
import { Quaternion } from '@src/math/Quaternion.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Vector4 } from '@src/math/Vector4.js';
import { Euler } from '@src/math/Euler.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import {
	x,
	y,
	z,
	w,
	eps
} from '@test-utils/math-constants.js';

const orders = [ 'XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY' ];
const eulerAngles = new Euler( 0.1, - 0.3, 0.25 );

function qSub( a, b ) {

	const result = new Quaternion();
	result.copy( a );

	result.x -= b.x;
	result.y -= b.y;
	result.z -= b.z;
	result.w -= b.w;

	return result;

}

function doSlerpObject( aArr, bArr, t ) {

	const a = new Quaternion().fromArray( aArr ),
		b = new Quaternion().fromArray( bArr ),
		c = new Quaternion().fromArray( aArr );

	c.slerp( b, t );

	return {

		equals: function ( x, y, z, w, maxError ) {

			if ( maxError === undefined ) maxError = Number.EPSILON;

			return Math.abs( x - c.x ) <= maxError &&
				Math.abs( y - c.y ) <= maxError &&
				Math.abs( z - c.z ) <= maxError &&
				Math.abs( w - c.w ) <= maxError;

		},

		length: c.length(),

		dotA: c.dot( a ),
		dotB: c.dot( b )

	};

}

function doSlerpArray( a, b, t ) {

	const result = [ 0, 0, 0, 0 ];

	Quaternion.slerpFlat( result, 0, a, 0, b, 0, t );

	function arrDot( a, b ) {

		return a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] +
			a[ 2 ] * b[ 2 ] + a[ 3 ] * b[ 3 ];

	}

	return {

		equals: function ( x, y, z, w, maxError ) {

			if ( maxError === undefined ) maxError = Number.EPSILON;

			return Math.abs( x - result[ 0 ] ) <= maxError &&
				Math.abs( y - result[ 1 ] ) <= maxError &&
				Math.abs( z - result[ 2 ] ) <= maxError &&
				Math.abs( w - result[ 3 ] ) <= maxError;

		},

		length: Math.sqrt( arrDot( result, result ) ),

		dotA: arrDot( result, a ),
		dotB: arrDot( result, b )

	};

}

function slerpTestSkeleton( doSlerp, maxError ) {

	let result;

	const a = [
		0.6753410084407496,
		0.4087830051091744,
		0.32856700410659473,
		0.5185120064806223
	];

	const b = [
		0.6602792107657797,
		0.43647413932562285,
		0.35119011210236006,
		0.5001871596632682
	];

	let maxNormError = 0;

	function isNormal( result ) {

		const normError = Math.abs( 1 - result.length );
		maxNormError = Math.max( maxNormError, normError );
		return normError <= maxError;

	}

	result = doSlerp( a, b, 0 );
	expect( result.equals(
		a[ 0 ], a[ 1 ], a[ 2 ], a[ 3 ], 0 ) ).toBeTruthy();

	result = doSlerp( a, b, 1 );
	expect( result.equals(
		b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ], 0 ) ).toBeTruthy();

	result = doSlerp( a, b, 0.5 );
	expect( Math.abs( result.dotA - result.dotB ) <= Number.EPSILON ).toBeTruthy();
	expect( isNormal( result ) ).toBeTruthy();

	result = doSlerp( a, b, 0.25 );
	expect( result.dotA > result.dotB ).toBeTruthy();
	expect( isNormal( result ) ).toBeTruthy();

	result = doSlerp( a, b, 0.75 );
	expect( result.dotA < result.dotB ).toBeTruthy();
	expect( isNormal( result ) ).toBeTruthy();

	const D = Math.SQRT1_2;

	result = doSlerp( [ 1, 0, 0, 0 ], [ 0, 0, 1, 0 ], 0.5 );
	expect( result.equals( D, 0, D, 0 ) ).toBeTruthy();
	expect( isNormal( result ) ).toBeTruthy();

	result = doSlerp( [ 0, D, 0, D ], [ 0, - D, 0, D ], 0.5 );
	expect( result.equals( 0, 0, 0, 1 ) ).toBeTruthy();
	expect( isNormal( result ) ).toBeTruthy();

}

function changeEulerOrder( euler, order ) {

	return new Euler( euler.x, euler.y, euler.z, order );

}

describe( 'Maths', () => {

	describe( 'Quaternion', () => {

		test( 'Instancing', () => {

			let a = new Quaternion();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 1 ).toBeTruthy();

			a = new Quaternion( x, y, z, w );
			expect( a.x === x ).toBeTruthy();
			expect( a.y === y ).toBeTruthy();
			expect( a.z === z ).toBeTruthy();
			expect( a.w === w ).toBeTruthy();

		} );

		test( 'slerp', () => {

			slerpTestSkeleton( doSlerpObject, Number.EPSILON );

		} );

		test( 'slerpFlat', () => {

			slerpTestSkeleton( doSlerpArray, Number.EPSILON );

		} );

		test( 'properties', () => {

			expect.assertions( 8 );

			const a = new Quaternion();
			a._onChange( function () {

				expect( true ).toBeTruthy();

			} );

			a.x = x;
			a.y = y;
			a.z = z;
			a.w = w;

			expect( a.x ).toBe( x );
			expect( a.y ).toBe( y );
			expect( a.z ).toBe( z );
			expect( a.w ).toBe( w );

		} );

		test( 'x', () => {

			let a = new Quaternion();
			expect( a.x === 0 ).toBeTruthy();

			a = new Quaternion( 1, 2, 3 );
			expect( a.x === 1 ).toBeTruthy();

			a = new Quaternion( 4, 5, 6, 1 );
			expect( a.x === 4 ).toBeTruthy();

			a = new Quaternion( 7, 8, 9 );
			a.x = 10;
			expect( a.x === 10 ).toBeTruthy();

			a = new Quaternion( 11, 12, 13 );
			let b = false;
			a._onChange( function () {

				b = true;

			} );
			expect( ! b ).toBeTruthy();
			a.x = 14;
			expect( b ).toBeTruthy();
			expect( a.x === 14 ).toBeTruthy();

		} );

		test( 'y', () => {

			let a = new Quaternion();
			expect( a.y === 0 ).toBeTruthy();

			a = new Quaternion( 1, 2, 3 );
			expect( a.y === 2 ).toBeTruthy();

			a = new Quaternion( 4, 5, 6, 1 );
			expect( a.y === 5 ).toBeTruthy();

			a = new Quaternion( 7, 8, 9 );
			a.y = 10;
			expect( a.y === 10 ).toBeTruthy();

			a = new Quaternion( 11, 12, 13 );
			let b = false;
			a._onChange( function () {

				b = true;

			} );
			expect( ! b ).toBeTruthy();
			a.y = 14;
			expect( b ).toBeTruthy();
			expect( a.y === 14 ).toBeTruthy();

		} );

		test( 'z', () => {

			let a = new Quaternion();
			expect( a.z === 0 ).toBeTruthy();

			a = new Quaternion( 1, 2, 3 );
			expect( a.z === 3 ).toBeTruthy();

			a = new Quaternion( 4, 5, 6, 1 );
			expect( a.z === 6 ).toBeTruthy();

			a = new Quaternion( 7, 8, 9 );
			a.z = 10;
			expect( a.z === 10 ).toBeTruthy();

			a = new Quaternion( 11, 12, 13 );
			let b = false;
			a._onChange( function () {

				b = true;

			} );
			expect( ! b ).toBeTruthy();
			a.z = 14;
			expect( b ).toBeTruthy();
			expect( a.z === 14 ).toBeTruthy();

		} );

		test( 'w', () => {

			let a = new Quaternion();
			expect( a.w === 1 ).toBeTruthy();

			a = new Quaternion( 1, 2, 3 );
			expect( a.w === 1 ).toBeTruthy();

			a = new Quaternion( 4, 5, 6, 1 );
			expect( a.w === 1 ).toBeTruthy();

			a = new Quaternion( 7, 8, 9 );
			a.w = 10;
			expect( a.w === 10 ).toBeTruthy();

			a = new Quaternion( 11, 12, 13 );
			let b = false;
			a._onChange( function () {

				b = true;

			} );
			expect( ! b ).toBeTruthy();
			a.w = 14;
			expect( b ).toBeTruthy();
			expect( a.w === 14 ).toBeTruthy();

		} );

		test( 'isQuaternion', () => {

			const object = new Quaternion();
			expect( object.isQuaternion ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Quaternion();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 1 ).toBeTruthy();

			a.set( x, y, z, w );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();
			expect( a.z === z ).toBeTruthy();
			expect( a.w === w ).toBeTruthy();

		} );

		test( 'clone', () => {

			const a = new Quaternion().clone();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 1 ).toBeTruthy();

			const b = a.set( x, y, z, w ).clone();
			expect( b.x == x ).toBeTruthy();
			expect( b.y == y ).toBeTruthy();
			expect( b.z === z ).toBeTruthy();
			expect( b.w === w ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Quaternion( x, y, z, w );
			const b = new Quaternion().copy( a );
			expect( b.x == x ).toBeTruthy();
			expect( b.y == y ).toBeTruthy();
			expect( b.z == z ).toBeTruthy();
			expect( b.w == w ).toBeTruthy();

			// ensure that it is a true copy
			a.x = 0;
			a.y = - 1;
			a.z = 0;
			a.w = - 1;
			expect( b.x == x ).toBeTruthy();
			expect( b.y == y ).toBeTruthy();

		} );

		test( 'setFromEuler/setFromQuaternion', () => {

			const angles = [ new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ), new Vector3( 0, 0, 1 ) ];

			// ensure euler conversion to/from Quaternion matches.
			for ( let i = 0; i < orders.length; i ++ ) {

				for ( let j = 0; j < angles.length; j ++ ) {

					const eulers2 = new Euler().setFromQuaternion( new Quaternion().setFromEuler( new Euler( angles[ j ].x, angles[ j ].y, angles[ j ].z, orders[ i ] ) ), orders[ i ] );
					const newAngle = new Vector3( eulers2.x, eulers2.y, eulers2.z );
					expect( newAngle.distanceTo( angles[ j ] ) < 0.001 ).toBeTruthy();

				}

			}

		} );

		test( 'setFromAxisAngle', () => {

			// TODO: find cases to validate.

			const zero = new Quaternion();

			let a = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), 0 );
			expect( a.equals( zero ) ).toBeTruthy();
			a = new Quaternion().setFromAxisAngle( new Vector3( 0, 1, 0 ), 0 );
			expect( a.equals( zero ) ).toBeTruthy();
			a = new Quaternion().setFromAxisAngle( new Vector3( 0, 0, 1 ), 0 );
			expect( a.equals( zero ) ).toBeTruthy();

			const b1 = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), Math.PI );
			expect( ! a.equals( b1 ) ).toBeTruthy();
			const b2 = new Quaternion().setFromAxisAngle( new Vector3( 1, 0, 0 ), - Math.PI );
			expect( ! a.equals( b2 ) ).toBeTruthy();

			b1.multiply( b2 );
			expect( a.equals( b1 ) ).toBeTruthy();

		} );

		test( 'setFromEuler/setFromRotationMatrix', () => {

			// ensure euler conversion for Quaternion matches that of Matrix4
			for ( let i = 0; i < orders.length; i ++ ) {

				const q = new Quaternion().setFromEuler( changeEulerOrder( eulerAngles, orders[ i ] ) );
				const m = new Matrix4().makeRotationFromEuler( changeEulerOrder( eulerAngles, orders[ i ] ) );
				const q2 = new Quaternion().setFromRotationMatrix( m );

				expect( qSub( q, q2 ).length() < 0.001 ).toBeTruthy();

			}

		} );

		test( 'setFromRotationMatrix', () => {

			// contrived examples purely to please the god of code coverage...
			// match conditions in various 'else [if]' blocks

			const a = new Quaternion();
			let q = new Quaternion( - 9, - 2, 3, - 4 ).normalize();
			const m = new Matrix4().makeRotationFromQuaternion( q );
			let expected = new Vector4( 0.8581163303210332, 0.19069251784911848, - 0.2860387767736777, 0.38138503569823695 );

			a.setFromRotationMatrix( m );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

			q = new Quaternion( - 1, - 2, 1, - 1 ).normalize();
			m.makeRotationFromQuaternion( q );
			expected = new Vector4( 0.37796447300922714, 0.7559289460184544, - 0.37796447300922714, 0.37796447300922714 );

			a.setFromRotationMatrix( m );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

		} );

		test( 'setFromUnitVectors', () => {

			const a = new Quaternion();
			const b = new Vector3( 1, 0, 0 );
			const c = new Vector3( 0, 1, 0 );
			const expected = new Quaternion( 0, 0, Math.sqrt( 2 ) / 2, Math.sqrt( 2 ) / 2 );

			a.setFromUnitVectors( b, c );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

		} );

		test( 'angleTo', () => {

			const a = new Quaternion();
			const b = new Quaternion().setFromEuler( new Euler( 0, Math.PI, 0 ) );
			const c = new Quaternion().setFromEuler( new Euler( 0, Math.PI * 2, 0 ) );

			expect( a.angleTo( a ) === 0 ).toBeTruthy();
			expect( a.angleTo( b ) === Math.PI ).toBeTruthy();
			expect( a.angleTo( c ) === 0 ).toBeTruthy();

		} );

		test( 'rotateTowards', () => {

			const a = new Quaternion();
			const b = new Quaternion().setFromEuler( new Euler( 0, Math.PI, 0 ) );
			const c = new Quaternion();

			const halfPI = Math.PI * 0.5;

			a.rotateTowards( b, 0 );
			expect( a.equals( a ) === true ).toBeTruthy();

			a.rotateTowards( b, Math.PI * 2 ); // test overshoot
			expect( a.equals( b ) === true ).toBeTruthy();

			a.set( 0, 0, 0, 1 );
			a.rotateTowards( b, halfPI );
			expect( a.angleTo( c ) - halfPI <= eps ).toBeTruthy();

		} );

		test( 'identity', () => {

			const a = new Quaternion();

			a.set( x, y, z, w );
			a.identity();

			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z === 0 ).toBeTruthy();
			expect( a.w === 1 ).toBeTruthy();

		} );

		test( 'invert/conjugate', () => {

			const a = new Quaternion( x, y, z, w );

			// TODO: add better validation here.

			const b = a.clone().conjugate();

			expect( a.x == - b.x ).toBeTruthy();
			expect( a.y == - b.y ).toBeTruthy();
			expect( a.z == - b.z ).toBeTruthy();
			expect( a.w == b.w ).toBeTruthy();

		} );

		test( 'dot', () => {

			let a = new Quaternion();
			let b = new Quaternion();

			expect( a.dot( b ) === 1 ).toBeTruthy();
			a = new Quaternion( 1, 2, 3, 1 );
			b = new Quaternion( 3, 2, 1, 1 );

			expect( a.dot( b ) === 11 ).toBeTruthy();

		} );

		test( 'normalize/length/lengthSq', () => {

			const a = new Quaternion( x, y, z, w );

			expect( a.length() != 1 ).toBeTruthy();
			expect( a.lengthSq() != 1 ).toBeTruthy();
			a.normalize();
			expect( a.length() == 1 ).toBeTruthy();
			expect( a.lengthSq() == 1 ).toBeTruthy();

			a.set( 0, 0, 0, 0 );
			expect( a.lengthSq() == 0 ).toBeTruthy();
			expect( a.length() == 0 ).toBeTruthy();
			a.normalize();
			expect( a.lengthSq() == 1 ).toBeTruthy();
			expect( a.length() == 1 ).toBeTruthy();

		} );

		test( 'multiplyQuaternions/multiply', () => {

			const angles = [ new Euler( 1, 0, 0 ), new Euler( 0, 1, 0 ), new Euler( 0, 0, 1 ) ];

			const q1 = new Quaternion().setFromEuler( changeEulerOrder( angles[ 0 ], 'XYZ' ) );
			const q2 = new Quaternion().setFromEuler( changeEulerOrder( angles[ 1 ], 'XYZ' ) );
			const q3 = new Quaternion().setFromEuler( changeEulerOrder( angles[ 2 ], 'XYZ' ) );

			const q = new Quaternion().multiplyQuaternions( q1, q2 ).multiply( q3 );

			const m1 = new Matrix4().makeRotationFromEuler( changeEulerOrder( angles[ 0 ], 'XYZ' ) );
			const m2 = new Matrix4().makeRotationFromEuler( changeEulerOrder( angles[ 1 ], 'XYZ' ) );
			const m3 = new Matrix4().makeRotationFromEuler( changeEulerOrder( angles[ 2 ], 'XYZ' ) );

			const m = new Matrix4().multiplyMatrices( m1, m2 ).multiply( m3 );

			const qFromM = new Quaternion().setFromRotationMatrix( m );

			expect( qSub( q, qFromM ).length() < 0.001 ).toBeTruthy();

		} );

		test( 'premultiply', () => {

			const a = new Quaternion( x, y, z, w );
			const b = new Quaternion( 2 * x, - y, - 2 * z, w );
			const expected = new Quaternion( 42, - 32, - 2, 58 );

			a.premultiply( b );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

		} );

		test( 'slerp', () => {

			const a = new Quaternion( x, y, z, w ).normalize();
			const b = new Quaternion( w, x, y, z ).normalize();

			const c = a.clone().slerp( b, 0 );
			const d = a.clone().slerp( b, 1 );

			expect( a.equals( c ) ).toBeTruthy();
			expect( b.equals( d ) ).toBeTruthy();

			const D = Math.SQRT1_2;

			const e = new Quaternion( 1, 0, 0, 0 );
			const f = new Quaternion( 0, 0, 1, 0 );
			let expected = new Quaternion( D, 0, D, 0 );
			let result = e.clone().slerp( f, 0.5 );
			expect( Math.abs( result.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( result.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( result.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( result.w - expected.w ) <= eps ).toBeTruthy();

			const g = new Quaternion( 0, D, 0, D );
			const h = new Quaternion( 0, - D, 0, D );
			expected = new Quaternion( 0, 0, 0, 1 );
			result = g.clone().slerp( h, 0.5 );

			expect( Math.abs( result.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( result.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( result.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( result.w - expected.w ) <= eps ).toBeTruthy();

		} );

		test( 'slerpQuaternions', () => {

			const e = new Quaternion( 1, 0, 0, 0 );
			const f = new Quaternion( 0, 0, 1, 0 );
			const expected = new Quaternion( Math.SQRT1_2, 0, Math.SQRT1_2, 0 );

			const a = new Quaternion();
			a.slerpQuaternions( e, f, 0.5 );

			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

		} );

		test( 'random', () => {

			const a = new Quaternion();

			a.random();

			const identity = new Quaternion();
			expect( a ).not.toEqualLikeQUnit( identity );

			expect( ( 1 - a.length() ) <= Number.EPSILON ).toBeTruthy();

		} );

		test( 'equals', () => {

			const a = new Quaternion( x, y, z, w );
			const b = new Quaternion( - x, - y, - z, - w );

			expect( a.x != b.x ).toBeTruthy();
			expect( a.y != b.y ).toBeTruthy();

			expect( ! a.equals( b ) ).toBeTruthy();
			expect( ! b.equals( a ) ).toBeTruthy();

			a.copy( b );
			expect( a.x == b.x ).toBeTruthy();
			expect( a.y == b.y ).toBeTruthy();

			expect( a.equals( b ) ).toBeTruthy();
			expect( b.equals( a ) ).toBeTruthy();

		} );

		test( 'fromArray', () => {

			const a = new Quaternion();
			a.fromArray( [ x, y, z, w ] );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();
			expect( a.z === z ).toBeTruthy();
			expect( a.w === w ).toBeTruthy();

			a.fromArray( [ undefined, x, y, z, w, undefined ], 1 );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();
			expect( a.z === z ).toBeTruthy();
			expect( a.w === w ).toBeTruthy();

		} );

		test( 'toArray', () => {

			const a = new Quaternion( x, y, z, w );

			let array = a.toArray();
			expect( array[ 0 ] ).toBe( x );
			expect( array[ 1 ] ).toBe( y );
			expect( array[ 2 ] ).toBe( z );
			expect( array[ 3 ] ).toBe( w );

			array = [];
			a.toArray( array );
			expect( array[ 0 ] ).toBe( x );
			expect( array[ 1 ] ).toBe( y );
			expect( array[ 2 ] ).toBe( z );
			expect( array[ 3 ] ).toBe( w );

			array = [];
			a.toArray( array, 1 );
			expect( array[ 0 ] ).toBe( undefined );
			expect( array[ 1 ] ).toBe( x );
			expect( array[ 2 ] ).toBe( y );
			expect( array[ 3 ] ).toBe( z );
			expect( array[ 4 ] ).toBe( w );

		} );

		test( 'fromBufferAttribute', () => {

			const a = new Quaternion();

			const attribute = new BufferAttribute( new Float32Array( [

				0, 0, 0, 1,
				.7, 0, 0, .7,
				0, .7, 0, .7,

			] ), 4 );

			a.fromBufferAttribute( attribute, 0 );
			expect( a.x ).toNumEqual( 0 );
			expect( a.y ).toNumEqual( 0 );
			expect( a.z ).toNumEqual( 0 );
			expect( a.w ).toNumEqual( 1 );

			a.fromBufferAttribute( attribute, 1 );
			expect( a.x ).toNumEqual( .7 );
			expect( a.y ).toNumEqual( 0 );
			expect( a.z ).toNumEqual( 0 );
			expect( a.w ).toNumEqual( .7 );

			a.fromBufferAttribute( attribute, 2 );
			expect( a.x ).toNumEqual( 0 );
			expect( a.y ).toNumEqual( .7 );
			expect( a.z ).toNumEqual( 0 );
			expect( a.w ).toNumEqual( .7 );

		} );

		test( '_onChange', () => {

			let b = false;
			const f = function () {

				b = true;

			};

			const a = new Quaternion( 11, 12, 13, 1 );
			a._onChange( f );
			expect( a._onChangeCallback === f ).toBeTruthy();

			a._onChangeCallback();
			expect( b ).toBeTruthy();

		} );

		test( '_onChangeCallback', () => {

			let b = false;
			const a = new Quaternion( 11, 12, 13, 1 );
			const f = function () {

				b = true;
				expect( a === this ).toBeTruthy();

			};

			a._onChangeCallback = f;
			expect( a._onChangeCallback === f ).toBeTruthy();

			a._onChangeCallback();
			expect( b ).toBeTruthy();

		} );

		test( 'multiplyVector3', () => {

			const angles = [ new Euler( 1, 0, 0 ), new Euler( 0, 1, 0 ), new Euler( 0, 0, 1 ) ];

			// ensure euler conversion for Quaternion matches that of Matrix4
			for ( let i = 0; i < orders.length; i ++ ) {

				for ( let j = 0; j < angles.length; j ++ ) {

					const q = new Quaternion().setFromEuler( changeEulerOrder( angles[ j ], orders[ i ] ) );
					const m = new Matrix4().makeRotationFromEuler( changeEulerOrder( angles[ j ], orders[ i ] ) );

					const v0 = new Vector3( 1, 0, 0 );
					const qv = v0.clone().applyQuaternion( q );
					const mv = v0.clone().applyMatrix4( m );

					expect( qv.distanceTo( mv ) < 0.001 ).toBeTruthy();

				}

			}

		} );

		test( 'toJSON', () => {

			const q = new Quaternion( 0, 0.5, 0.7, 1 );
			const array = q.toJSON();
			expect( array[ 0 ] ).toBe( 0 );
			expect( array[ 1 ] ).toBe( 0.5 );
			expect( array[ 2 ] ).toBe( 0.7 );
			expect( array[ 3 ] ).toBe( 1 );

		} );

		test( 'iterable', () => {

			const q = new Quaternion( 0, 0.5, 0.7, 1 );
			const array = [ ...q ];
			expect( array[ 0 ] ).toBe( 0 );
			expect( array[ 1 ] ).toBe( 0.5 );
			expect( array[ 2 ] ).toBe( 0.7 );
			expect( array[ 3 ] ).toBe( 1 );

		} );

	} );

} );

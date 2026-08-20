import { describe, test, expect } from 'vitest';
import { Vector4 } from '@src/math/Vector4.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import { BufferAttribute } from '@src/core/BufferAttribute.js';
import {
	x,
	y,
	z,
	w,
	eps
} from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Vector4', () => {

		test( 'Instancing', () => {

			let a = new Vector4();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 1 ).toBeTruthy();

			a = new Vector4( x, y, z, w );
			expect( a.x === x ).toBeTruthy();
			expect( a.y === y ).toBeTruthy();
			expect( a.z === z ).toBeTruthy();
			expect( a.w === w ).toBeTruthy();

		} );

		test( 'isVector4', () => {

			const object = new Vector4();
			expect( object.isVector4 ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Vector4();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 1 ).toBeTruthy();

			a.set( x, y, z, w );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();
			expect( a.z == z ).toBeTruthy();
			expect( a.w == w ).toBeTruthy();

		} );

		test( 'setX', () => {

			const a = new Vector4();
			expect( a.x == 0 ).toBeTruthy();

			a.setX( x );
			expect( a.x == x ).toBeTruthy();

		} );

		test( 'setY', () => {

			const a = new Vector4();
			expect( a.y == 0 ).toBeTruthy();

			a.setY( y );
			expect( a.y == y ).toBeTruthy();

		} );

		test( 'setZ', () => {

			const a = new Vector4();
			expect( a.z == 0 ).toBeTruthy();

			a.setZ( z );
			expect( a.z == z ).toBeTruthy();

		} );

		test( 'setW', () => {

			const a = new Vector4();
			expect( a.w == 1 ).toBeTruthy();

			a.setW( w );
			expect( a.w == w ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Vector4( x, y, z, w );
			const b = new Vector4().copy( a );
			expect( b.x == x ).toBeTruthy();
			expect( b.y == y ).toBeTruthy();
			expect( b.z == z ).toBeTruthy();
			expect( b.w == w ).toBeTruthy();

			// ensure that it is a true copy
			a.x = 0;
			a.y = - 1;
			a.z = - 2;
			a.w = - 3;
			expect( b.x == x ).toBeTruthy();
			expect( b.y == y ).toBeTruthy();
			expect( b.z == z ).toBeTruthy();
			expect( b.w == w ).toBeTruthy();

		} );

		test( 'add', () => {

			const a = new Vector4( x, y, z, w );
			const b = new Vector4( - x, - y, - z, - w );

			a.add( b );
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 0 ).toBeTruthy();

		} );

		test( 'addVectors', () => {

			const b = new Vector4( - x, - y, - z, - w );
			const c = new Vector4().addVectors( b, b );

			expect( c.x == - 2 * x ).toBeTruthy();
			expect( c.y == - 2 * y ).toBeTruthy();
			expect( c.z == - 2 * z ).toBeTruthy();
			expect( c.w == - 2 * w ).toBeTruthy();

		} );

		test( 'addScaledVector', () => {

			const a = new Vector4( x, y, z, w );
			const b = new Vector4( 6, 7, 8, 9 );
			const s = 3;

			a.addScaledVector( b, s );
			expect( a.x ).toBe( x + b.x * s );
			expect( a.y ).toBe( y + b.y * s );
			expect( a.z ).toBe( z + b.z * s );
			expect( a.w ).toBe( w + b.w * s );

		} );

		test( 'sub', () => {

			const a = new Vector4( x, y, z, w );
			const b = new Vector4( - x, - y, - z, - w );

			a.sub( b );
			expect( a.x == 2 * x ).toBeTruthy();
			expect( a.y == 2 * y ).toBeTruthy();
			expect( a.z == 2 * z ).toBeTruthy();
			expect( a.w == 2 * w ).toBeTruthy();

		} );

		test( 'subVectors', () => {

			const a = new Vector4( x, y, z, w );
			const c = new Vector4().subVectors( a, a );
			expect( c.x == 0 ).toBeTruthy();
			expect( c.y == 0 ).toBeTruthy();
			expect( c.z == 0 ).toBeTruthy();
			expect( c.w == 0 ).toBeTruthy();

		} );

		test( 'applyMatrix4', () => {

			const a = new Vector4( x, y, z, w );
			const m = new Matrix4().makeRotationX( Math.PI );
			const expected = new Vector4( 2, - 3, - 4, 5 );

			a.applyMatrix4( m );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

			a.set( x, y, z, w );
			m.makeTranslation( 5, 7, 11 );
			expected.set( 27, 38, 59, 5 );

			a.applyMatrix4( m );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

			a.set( x, y, z, w );
			m.set( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 );
			expected.set( 2, 3, 4, 4 );

			a.applyMatrix4( m );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

			a.set( x, y, z, w );
			m.set( 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 );
			expected.set( 68, 224, 442, 664 );

			a.applyMatrix4( m );
			expect( Math.abs( a.x - expected.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - expected.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - expected.w ) <= eps ).toBeTruthy();

		} );

		test( 'divide', () => {

			const a = new Vector4( 7, 8, 9, 0 );
			const b = new Vector4( 2, 2, 3, 4 );

			a.divide( b );
			expect( a.x ).toBe( 3.5 );
			expect( a.y ).toBe( 4.0 );
			expect( a.z ).toBe( 3.0 );
			expect( a.w ).toBe( 0.0 );

		} );

		test( 'setFromMatrixPosition', () => {

			const a = new Vector4();
			const m = new Matrix4().set( 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 );

			a.setFromMatrixPosition( m );
			expect( a.x ).toBe( 7 );
			expect( a.y ).toBe( 19 );
			expect( a.z ).toBe( 37 );
			expect( a.w ).toBe( 53 );

		} );

		test( 'clampScalar', () => {

			const a = new Vector4( - 0.1, 0.01, 0.5, 1.5 );
			const clamped = new Vector4( 0.1, 0.1, 0.5, 1.0 );

			a.clampScalar( 0.1, 1.0 );
			expect( Math.abs( a.x - clamped.x ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - clamped.y ) <= eps ).toBeTruthy();
			expect( Math.abs( a.z - clamped.z ) <= eps ).toBeTruthy();
			expect( Math.abs( a.w - clamped.w ) <= eps ).toBeTruthy();

		} );

		test( 'negate', () => {

			const a = new Vector4( x, y, z, w );

			a.negate();
			expect( a.x == - x ).toBeTruthy();
			expect( a.y == - y ).toBeTruthy();
			expect( a.z == - z ).toBeTruthy();
			expect( a.w == - w ).toBeTruthy();

		} );

		test( 'dot', () => {

			const a = new Vector4( x, y, z, w );
			const b = new Vector4( - x, - y, - z, - w );
			const c = new Vector4( 0, 0, 0, 0 );

			let result = a.dot( b );
			expect( result == ( - x * x - y * y - z * z - w * w ) ).toBeTruthy();

			result = a.dot( c );
			expect( result == 0 ).toBeTruthy();

		} );

		test( 'manhattanLength', () => {

			const a = new Vector4( x, 0, 0, 0 );
			const b = new Vector4( 0, - y, 0, 0 );
			const c = new Vector4( 0, 0, z, 0 );
			const d = new Vector4( 0, 0, 0, w );
			const e = new Vector4( 0, 0, 0, 0 );

			expect( a.manhattanLength() == x ).toBeTruthy();
			expect( b.manhattanLength() == y ).toBeTruthy();
			expect( c.manhattanLength() == z ).toBeTruthy();
			expect( d.manhattanLength() == w ).toBeTruthy();
			expect( e.manhattanLength() == 0 ).toBeTruthy();

			a.set( x, y, z, w );
			expect(
				a.manhattanLength() == Math.abs( x ) + Math.abs( y ) + Math.abs( z ) + Math.abs( w )
			).toBeTruthy();

		} );

		test( 'normalize', () => {

			const a = new Vector4( x, 0, 0, 0 );
			const b = new Vector4( 0, - y, 0, 0 );
			const c = new Vector4( 0, 0, z, 0 );
			const d = new Vector4( 0, 0, 0, - w );

			a.normalize();
			expect( a.length() == 1 ).toBeTruthy();
			expect( a.x == 1 ).toBeTruthy();

			b.normalize();
			expect( b.length() == 1 ).toBeTruthy();
			expect( b.y == - 1 ).toBeTruthy();

			c.normalize();
			expect( c.length() == 1 ).toBeTruthy();
			expect( c.z == 1 ).toBeTruthy();

			d.normalize();
			expect( d.length() == 1 ).toBeTruthy();
			expect( d.w == - 1 ).toBeTruthy();

		} );

		test( 'setLength', () => {

			let a = new Vector4( x, 0, 0, 0 );

			expect( a.length() == x ).toBeTruthy();
			a.setLength( y );
			expect( a.length() == y ).toBeTruthy();

			a = new Vector4( 0, 0, 0, 0 );
			expect( a.length() == 0 ).toBeTruthy();
			a.setLength( y );
			expect( a.length() == 0 ).toBeTruthy();
			a.setLength();
			expect( isNaN( a.length() ) ).toBeTruthy();

		} );

		test( 'equals', () => {

			const a = new Vector4( x, 0, z, 0 );
			const b = new Vector4( 0, - y, 0, - w );

			expect( a.x != b.x ).toBeTruthy();
			expect( a.y != b.y ).toBeTruthy();
			expect( a.z != b.z ).toBeTruthy();
			expect( a.w != b.w ).toBeTruthy();

			expect( ! a.equals( b ) ).toBeTruthy();
			expect( ! b.equals( a ) ).toBeTruthy();

			a.copy( b );
			expect( a.x == b.x ).toBeTruthy();
			expect( a.y == b.y ).toBeTruthy();
			expect( a.z == b.z ).toBeTruthy();
			expect( a.w == b.w ).toBeTruthy();

			expect( a.equals( b ) ).toBeTruthy();
			expect( b.equals( a ) ).toBeTruthy();

		} );

		test( 'fromArray', () => {

			const a = new Vector4();
			const array = [ 1, 2, 3, 4, 5, 6, 7, 8 ];

			a.fromArray( array );
			expect( a.x ).toBe( 1 );
			expect( a.y ).toBe( 2 );
			expect( a.z ).toBe( 3 );
			expect( a.w ).toBe( 4 );

			a.fromArray( array, 4 );
			expect( a.x ).toBe( 5 );
			expect( a.y ).toBe( 6 );
			expect( a.z ).toBe( 7 );
			expect( a.w ).toBe( 8 );

		} );

		test( 'toArray', () => {

			const a = new Vector4( x, y, z, w );

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

			const a = new Vector4();
			const attr = new BufferAttribute( new Float32Array( [ 1, 2, 3, 4, 5, 6, 7, 8 ] ), 4 );

			a.fromBufferAttribute( attr, 0 );
			expect( a.x ).toBe( 1 );
			expect( a.y ).toBe( 2 );
			expect( a.z ).toBe( 3 );
			expect( a.w ).toBe( 4 );

			a.fromBufferAttribute( attr, 1 );
			expect( a.x ).toBe( 5 );
			expect( a.y ).toBe( 6 );
			expect( a.z ).toBe( 7 );
			expect( a.w ).toBe( 8 );

		} );

		// TODO (Itee) refactor/split
		test( 'setX,setY,setZ,setW', () => {

			const a = new Vector4();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 1 ).toBeTruthy();

			a.setX( x );
			a.setY( y );
			a.setZ( z );
			a.setW( w );

			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();
			expect( a.z == z ).toBeTruthy();
			expect( a.w == w ).toBeTruthy();

		} );

		test( 'setComponent,getComponent', () => {

			const a = new Vector4();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();
			expect( a.z == 0 ).toBeTruthy();
			expect( a.w == 1 ).toBeTruthy();

			a.setComponent( 0, 1 );
			a.setComponent( 1, 2 );
			a.setComponent( 2, 3 );
			a.setComponent( 3, 4 );
			expect( a.getComponent( 0 ) == 1 ).toBeTruthy();
			expect( a.getComponent( 1 ) == 2 ).toBeTruthy();
			expect( a.getComponent( 2 ) == 3 ).toBeTruthy();
			expect( a.getComponent( 3 ) == 4 ).toBeTruthy();

		} );

		test( 'setComponent/getComponent exceptions', () => {

			const a = new Vector4();

			expect( () => {

				a.setComponent( 4, 0 );

			} ).toThrow( /index is out of range/ );

			expect( () => {

				a.getComponent( 4 );

			} ).toThrow( /index is out of range/ );

		} );

		test( 'setScalar/addScalar/subScalar', () => {

			const a = new Vector4();
			const s = 3;

			a.setScalar( s );
			expect( a.x ).toBe( s );
			expect( a.y ).toBe( s );
			expect( a.z ).toBe( s );
			expect( a.w ).toBe( s );

			a.addScalar( s );
			expect( a.x ).toBe( 2 * s );
			expect( a.y ).toBe( 2 * s );
			expect( a.z ).toBe( 2 * s );
			expect( a.w ).toBe( 2 * s );

			a.subScalar( 2 * s );
			expect( a.x ).toBe( 0 );
			expect( a.y ).toBe( 0 );
			expect( a.z ).toBe( 0 );
			expect( a.w ).toBe( 0 );

		} );

		test( 'multiplyScalar/divideScalar', () => {

			const a = new Vector4( x, y, z, w );
			const b = new Vector4( - x, - y, - z, - w );

			a.multiplyScalar( - 2 );
			expect( a.x == x * - 2 ).toBeTruthy();
			expect( a.y == y * - 2 ).toBeTruthy();
			expect( a.z == z * - 2 ).toBeTruthy();
			expect( a.w == w * - 2 ).toBeTruthy();

			b.multiplyScalar( - 2 );
			expect( b.x == 2 * x ).toBeTruthy();
			expect( b.y == 2 * y ).toBeTruthy();
			expect( b.z == 2 * z ).toBeTruthy();
			expect( b.w == 2 * w ).toBeTruthy();

			a.divideScalar( - 2 );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();
			expect( a.z == z ).toBeTruthy();
			expect( a.w == w ).toBeTruthy();

			b.divideScalar( - 2 );
			expect( b.x == - x ).toBeTruthy();
			expect( b.y == - y ).toBeTruthy();
			expect( b.z == - z ).toBeTruthy();
			expect( b.w == - w ).toBeTruthy();

		} );

		test( 'min/max/clamp', () => {

			const a = new Vector4( x, y, z, w );
			const b = new Vector4( - x, - y, - z, - w );
			const c = new Vector4();

			c.copy( a ).min( b );
			expect( c.x == - x ).toBeTruthy();
			expect( c.y == - y ).toBeTruthy();
			expect( c.z == - z ).toBeTruthy();
			expect( c.w == - w ).toBeTruthy();

			c.copy( a ).max( b );
			expect( c.x == x ).toBeTruthy();
			expect( c.y == y ).toBeTruthy();
			expect( c.z == z ).toBeTruthy();
			expect( c.w == w ).toBeTruthy();

			c.set( - 2 * x, 2 * y, - 2 * z, 2 * w );
			c.clamp( b, a );
			expect( c.x == - x ).toBeTruthy();
			expect( c.y == y ).toBeTruthy();
			expect( c.z == - z ).toBeTruthy();
			expect( c.w == w ).toBeTruthy();

		} );

		test( 'length/lengthSq', () => {

			const a = new Vector4( x, 0, 0, 0 );
			const b = new Vector4( 0, - y, 0, 0 );
			const c = new Vector4( 0, 0, z, 0 );
			const d = new Vector4( 0, 0, 0, w );
			const e = new Vector4( 0, 0, 0, 0 );

			expect( a.length() == x ).toBeTruthy();
			expect( a.lengthSq() == x * x ).toBeTruthy();
			expect( b.length() == y ).toBeTruthy();
			expect( b.lengthSq() == y * y ).toBeTruthy();
			expect( c.length() == z ).toBeTruthy();
			expect( c.lengthSq() == z * z ).toBeTruthy();
			expect( d.length() == w ).toBeTruthy();
			expect( d.lengthSq() == w * w ).toBeTruthy();
			expect( e.length() == 0 ).toBeTruthy();
			expect( e.lengthSq() == 0 ).toBeTruthy();

			a.set( x, y, z, w );
			expect( a.length() == Math.sqrt( x * x + y * y + z * z + w * w ) ).toBeTruthy();
			expect( a.lengthSq() == ( x * x + y * y + z * z + w * w ) ).toBeTruthy();

		} );

		test( 'lerp/clone', () => {

			const a = new Vector4( x, 0, z, 0 );
			const b = new Vector4( 0, - y, 0, - w );

			expect( a.lerp( a, 0 ).equals( a.lerp( a, 0.5 ) ) ).toBeTruthy();
			expect( a.lerp( a, 0 ).equals( a.lerp( a, 1 ) ) ).toBeTruthy();

			expect( a.clone().lerp( b, 0 ).equals( a ) ).toBeTruthy();

			expect( a.clone().lerp( b, 0.5 ).x == x * 0.5 ).toBeTruthy();
			expect( a.clone().lerp( b, 0.5 ).y == - y * 0.5 ).toBeTruthy();
			expect( a.clone().lerp( b, 0.5 ).z == z * 0.5 ).toBeTruthy();
			expect( a.clone().lerp( b, 0.5 ).w == - w * 0.5 ).toBeTruthy();

			expect( a.clone().lerp( b, 1 ).equals( b ) ).toBeTruthy();

		} );

		test( 'iterable', () => {

			const v = new Vector4( 0, 0.3, 0.7, 1 );
			const array = [ ...v ];
			expect( array[ 0 ] ).toBe( 0 );
			expect( array[ 1 ] ).toBe( 0.3 );
			expect( array[ 2 ] ).toBe( 0.7 );
			expect( array[ 3 ] ).toBe( 1 );

		} );

	} );

} );

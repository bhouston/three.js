import { describe, test, expect } from 'vitest';
import { Vector2 } from '@src/math/Vector2.js';
import { Matrix3 } from '@src/math/Matrix3.js';
import { BufferAttribute } from '@src/core/BufferAttribute.js';
import {
	x,
	y,
	eps
} from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Vector2', () => {

		test( 'Instancing', () => {

			let a = new Vector2();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();

			a = new Vector2( x, y );
			expect( a.x === x ).toBeTruthy();
			expect( a.y === y ).toBeTruthy();

		} );

		// ( [Itee] WHAT ??? o_O )
		test( 'properties', () => {

			const a = new Vector2( 0, 0 );
			const width = 100;
			const height = 200;

			expect( a.width = width ).toBeTruthy();
			expect( a.height = height ).toBeTruthy();

			a.set( width, height );
			expect( a.width ).toBe( width );
			expect( a.height ).toBe( height );

		} );

		test( 'isVector2', () => {

			const object = new Vector2();
			expect( object.isVector2 ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Vector2();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();

			a.set( x, y );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Vector2( x, y );
			const b = new Vector2().copy( a );
			expect( b.x == x ).toBeTruthy();
			expect( b.y == y ).toBeTruthy();

			// ensure that it is a true copy
			a.x = 0;
			a.y = - 1;
			expect( b.x == x ).toBeTruthy();
			expect( b.y == y ).toBeTruthy();

		} );

		test( 'add', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( - x, - y );

			a.add( b );
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();

			const c = new Vector2().addVectors( b, b );
			expect( c.x == - 2 * x ).toBeTruthy();
			expect( c.y == - 2 * y ).toBeTruthy();

		} );

		test( 'addScaledVector', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( 2, 3 );
			const s = 3;

			a.addScaledVector( b, s );
			expect( a.x ).toBe( x + b.x * s );
			expect( a.y ).toBe( y + b.y * s );

		} );

		test( 'sub', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( - x, - y );

			a.sub( b );
			expect( a.x == 2 * x ).toBeTruthy();
			expect( a.y == 2 * y ).toBeTruthy();

			const c = new Vector2().subVectors( a, a );
			expect( c.x == 0 ).toBeTruthy();
			expect( c.y == 0 ).toBeTruthy();

		} );

		test( 'applyMatrix3', () => {

			const a = new Vector2( x, y );
			const m = new Matrix3().set( 2, 3, 5, 7, 11, 13, 17, 19, 23 );

			a.applyMatrix3( m );
			expect( a.x ).toBe( 18 );
			expect( a.y ).toBe( 60 );

		} );

		test( 'negate', () => {

			const a = new Vector2( x, y );

			a.negate();
			expect( a.x == - x ).toBeTruthy();
			expect( a.y == - y ).toBeTruthy();

		} );

		test( 'dot', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( - x, - y );
			const c = new Vector2();

			let result = a.dot( b );
			expect( result == ( - x * x - y * y ) ).toBeTruthy();

			result = a.dot( c );
			expect( result == 0 ).toBeTruthy();

		} );

		test( 'cross', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( 2 * x, - y );
			const answer = - 18;
			const crossed = a.cross( b );

			expect( Math.abs( answer - crossed ) <= eps ).toBeTruthy();

		} );

		test( 'manhattanLength', () => {

			const a = new Vector2( x, 0 );
			const b = new Vector2( 0, - y );
			const c = new Vector2();

			expect( a.manhattanLength() ).toBe( x );
			expect( b.manhattanLength() ).toBe( y );
			expect( c.manhattanLength() ).toBe( 0 );

			a.set( x, y );
			expect( a.manhattanLength() ).toBe( Math.abs( x ) + Math.abs( y ) );

		} );

		test( 'normalize', () => {

			const a = new Vector2( x, 0 );
			const b = new Vector2( 0, - y );

			a.normalize();
			expect( a.length() == 1 ).toBeTruthy();
			expect( a.x == 1 ).toBeTruthy();

			b.normalize();
			expect( b.length() == 1 ).toBeTruthy();
			expect( b.y == - 1 ).toBeTruthy();

		} );

		test( 'angleTo', () => {

			const a = new Vector2( - 0.18851655680720186, 0.9820700116639124 );
			const b = new Vector2( 0.18851655680720186, - 0.9820700116639124 );

			expect( a.angleTo( a ) ).toBe( 0 );
			expect( a.angleTo( b ) ).toBe( Math.PI );

			const x = new Vector2( 1, 0 );
			const y = new Vector2( 0, 1 );

			expect( x.angleTo( y ) ).toBe( Math.PI / 2 );
			expect( y.angleTo( x ) ).toBe( Math.PI / 2 );

			expect( Math.abs( x.angleTo( new Vector2( 1, 1 ) ) - ( Math.PI / 4 ) ) < 0.0000001 ).toBeTruthy();

		} );

		test( 'setLength', () => {

			let a = new Vector2( x, 0 );

			expect( a.length() == x ).toBeTruthy();
			a.setLength( y );
			expect( a.length() == y ).toBeTruthy();

			a = new Vector2( 0, 0 );
			expect( a.length() == 0 ).toBeTruthy();
			a.setLength( y );
			expect( a.length() == 0 ).toBeTruthy();
			a.setLength();
			expect( isNaN( a.length() ) ).toBeTruthy();

		} );

		test( 'equals', () => {

			const a = new Vector2( x, 0 );
			const b = new Vector2( 0, - y );

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

			const a = new Vector2();
			const array = [ 1, 2, 3, 4 ];

			a.fromArray( array );
			expect( a.x ).toBe( 1 );
			expect( a.y ).toBe( 2 );

			a.fromArray( array, 2 );
			expect( a.x ).toBe( 3 );
			expect( a.y ).toBe( 4 );

		} );

		test( 'toArray', () => {

			const a = new Vector2( x, y );

			let array = a.toArray();
			expect( array[ 0 ] ).toBe( x );
			expect( array[ 1 ] ).toBe( y );

			array = [];
			a.toArray( array );
			expect( array[ 0 ] ).toBe( x );
			expect( array[ 1 ] ).toBe( y );

			array = [];
			a.toArray( array, 1 );
			expect( array[ 0 ] ).toBe( undefined );
			expect( array[ 1 ] ).toBe( x );
			expect( array[ 2 ] ).toBe( y );

		} );

		test( 'fromBufferAttribute', () => {

			const a = new Vector2();
			const attr = new BufferAttribute( new Float32Array( [ 1, 2, 3, 4 ] ), 2 );

			a.fromBufferAttribute( attr, 0 );
			expect( a.x ).toBe( 1 );
			expect( a.y ).toBe( 2 );

			a.fromBufferAttribute( attr, 1 );
			expect( a.x ).toBe( 3 );
			expect( a.y ).toBe( 4 );

		} );

		// TODO (Itee) refactor/split
		test( 'setX,setY', () => {

			const a = new Vector2();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();

			a.setX( x );
			a.setY( y );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();

		} );

		test( 'setComponent,getComponent', () => {

			const a = new Vector2();
			expect( a.x == 0 ).toBeTruthy();
			expect( a.y == 0 ).toBeTruthy();

			a.setComponent( 0, 1 );
			a.setComponent( 1, 2 );
			expect( a.getComponent( 0 ) == 1 ).toBeTruthy();
			expect( a.getComponent( 1 ) == 2 ).toBeTruthy();

		} );

		test( 'multiply/divide', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( - x, - y );

			a.multiplyScalar( - 2 );
			expect( a.x == x * - 2 ).toBeTruthy();
			expect( a.y == y * - 2 ).toBeTruthy();

			b.multiplyScalar( - 2 );
			expect( b.x == 2 * x ).toBeTruthy();
			expect( b.y == 2 * y ).toBeTruthy();

			a.divideScalar( - 2 );
			expect( a.x == x ).toBeTruthy();
			expect( a.y == y ).toBeTruthy();

			b.divideScalar( - 2 );
			expect( b.x == - x ).toBeTruthy();
			expect( b.y == - y ).toBeTruthy();

		} );

		test( 'min/max/clamp', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( - x, - y );
			const c = new Vector2();

			c.copy( a ).min( b );
			expect( c.x == - x ).toBeTruthy();
			expect( c.y == - y ).toBeTruthy();

			c.copy( a ).max( b );
			expect( c.x == x ).toBeTruthy();
			expect( c.y == y ).toBeTruthy();

			c.set( - 2 * x, 2 * y );
			c.clamp( b, a );
			expect( c.x == - x ).toBeTruthy();
			expect( c.y == y ).toBeTruthy();

			c.set( - 2 * x, 2 * x );
			c.clampScalar( - x, x );
			expect( c.x ).toBe( - x );
			expect( c.y ).toBe( x );

		} );

		test( 'rounding', () => {

			expect( new Vector2( - 0.1, 0.1 ).floor() ).toEqualLikeQUnit( new Vector2( - 1, 0 ) );
			expect( new Vector2( - 0.5, 0.5 ).floor() ).toEqualLikeQUnit( new Vector2( - 1, 0 ) );
			expect( new Vector2( - 0.9, 0.9 ).floor() ).toEqualLikeQUnit( new Vector2( - 1, 0 ) );

			expect( new Vector2( - 0.1, 0.1 ).ceil() ).toEqualLikeQUnit( new Vector2( 0, 1 ) );
			expect( new Vector2( - 0.5, 0.5 ).ceil() ).toEqualLikeQUnit( new Vector2( 0, 1 ) );
			expect( new Vector2( - 0.9, 0.9 ).ceil() ).toEqualLikeQUnit( new Vector2( 0, 1 ) );

			expect( new Vector2( - 0.1, 0.1 ).round() ).toEqualLikeQUnit( new Vector2( 0, 0 ) );
			expect( new Vector2( - 0.5, 0.5 ).round() ).toEqualLikeQUnit( new Vector2( 0, 1 ) );
			expect( new Vector2( - 0.9, 0.9 ).round() ).toEqualLikeQUnit( new Vector2( - 1, 1 ) );

			expect( new Vector2( - 0.1, 0.1 ).roundToZero() ).toEqualLikeQUnit( new Vector2( 0, 0 ) );
			expect( new Vector2( - 0.5, 0.5 ).roundToZero() ).toEqualLikeQUnit( new Vector2( 0, 0 ) );
			expect( new Vector2( - 0.9, 0.9 ).roundToZero() ).toEqualLikeQUnit( new Vector2( 0, 0 ) );
			expect( new Vector2( - 1.1, 1.1 ).roundToZero() ).toEqualLikeQUnit( new Vector2( - 1, 1 ) );
			expect( new Vector2( - 1.5, 1.5 ).roundToZero() ).toEqualLikeQUnit( new Vector2( - 1, 1 ) );
			expect( new Vector2( - 1.9, 1.9 ).roundToZero() ).toEqualLikeQUnit( new Vector2( - 1, 1 ) );

		} );

		test( 'length/lengthSq', () => {

			const a = new Vector2( x, 0 );
			const b = new Vector2( 0, - y );
			const c = new Vector2();

			expect( a.length() == x ).toBeTruthy();
			expect( a.lengthSq() == x * x ).toBeTruthy();
			expect( b.length() == y ).toBeTruthy();
			expect( b.lengthSq() == y * y ).toBeTruthy();
			expect( c.length() == 0 ).toBeTruthy();
			expect( c.lengthSq() == 0 ).toBeTruthy();

			a.set( x, y );
			expect( a.length() == Math.sqrt( x * x + y * y ) ).toBeTruthy();
			expect( a.lengthSq() == ( x * x + y * y ) ).toBeTruthy();

		} );

		test( 'distanceTo/distanceToSquared', () => {

			const a = new Vector2( x, 0 );
			const b = new Vector2( 0, - y );
			const c = new Vector2();

			expect( a.distanceTo( c ) == x ).toBeTruthy();
			expect( a.distanceToSquared( c ) == x * x ).toBeTruthy();

			expect( b.distanceTo( c ) == y ).toBeTruthy();
			expect( b.distanceToSquared( c ) == y * y ).toBeTruthy();

		} );

		test( 'lerp/clone', () => {

			const a = new Vector2( x, 0 );
			const b = new Vector2( 0, - y );

			expect( a.lerp( a, 0 ).equals( a.lerp( a, 0.5 ) ) ).toBeTruthy();
			expect( a.lerp( a, 0 ).equals( a.lerp( a, 1 ) ) ).toBeTruthy();

			expect( a.clone().lerp( b, 0 ).equals( a ) ).toBeTruthy();

			expect( a.clone().lerp( b, 0.5 ).x == x * 0.5 ).toBeTruthy();
			expect( a.clone().lerp( b, 0.5 ).y == - y * 0.5 ).toBeTruthy();

			expect( a.clone().lerp( b, 1 ).equals( b ) ).toBeTruthy();

		} );

		test( 'setComponent/getComponent exceptions', () => {

			const a = new Vector2( 0, 0 );

			expect( () => {

				a.setComponent( 2, 0 );

			} ).toThrow( /index is out of range/ );

			expect( () => {

				a.getComponent( 2 );

			} ).toThrow( /index is out of range/ );

		} );

		test( 'setScalar/addScalar/subScalar', () => {

			const a = new Vector2( 1, 1 );
			const s = 3;

			a.setScalar( s );
			expect( a.x ).toBe( s );
			expect( a.y ).toBe( s );

			a.addScalar( s );
			expect( a.x ).toBe( 2 * s );
			expect( a.y ).toBe( 2 * s );

			a.subScalar( 2 * s );
			expect( a.x ).toBe( 0 );
			expect( a.y ).toBe( 0 );

		} );

		test( 'multiply/divide', () => {

			const a = new Vector2( x, y );
			const b = new Vector2( 2 * x, 2 * y );
			const c = new Vector2( 4 * x, 4 * y );

			a.multiply( b );
			expect( a.x ).toBe( x * b.x );
			expect( a.y ).toBe( y * b.y );

			b.divide( c );
			expect( b.x ).toBe( 0.5 );
			expect( b.y ).toBe( 0.5 );

		} );

		test( 'iterable', () => {

			const v = new Vector2( 0, 1 );
			const array = [ ...v ];
			expect( array[ 0 ] ).toBe( 0 );
			expect( array[ 1 ] ).toBe( 1 );

		} );

	} );

} );

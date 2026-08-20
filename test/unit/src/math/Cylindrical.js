import { describe, test, expect } from 'vitest';
import { Cylindrical } from '@src/math/Cylindrical.js';
import { Vector3 } from '@src/math/Vector3.js';
import { eps } from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Cylindrical', () => {

		test( 'Instancing', () => {

			let a = new Cylindrical();
			const radius = 10.0;
			const theta = Math.PI;
			const y = 5;

			expect( a.radius ).toBe( 1.0 );
			expect( a.theta ).toBe( 0 );
			expect( a.y ).toBe( 0 );

			a = new Cylindrical( radius, theta, y );
			expect( a.radius ).toBe( radius );
			expect( a.theta ).toBe( theta );
			expect( a.y ).toBe( y );

		} );

		test( 'set', () => {

			const a = new Cylindrical();
			const radius = 10.0;
			const theta = Math.PI;
			const y = 5;

			a.set( radius, theta, y );
			expect( a.radius ).toBe( radius );
			expect( a.theta ).toBe( theta );
			expect( a.y ).toBe( y );

		} );

		test( 'clone', () => {

			const radius = 10.0;
			const theta = Math.PI;
			const y = 5;
			const a = new Cylindrical( radius, theta, y );
			const b = a.clone();

			expect( a ).toEqualLikeQUnit( b );

			a.radius = 1;
			expect( a ).not.toEqualLikeQUnit( b );

		} );

		test( 'copy', () => {

			const radius = 10.0;
			const theta = Math.PI;
			const y = 5;
			const a = new Cylindrical( radius, theta, y );
			const b = new Cylindrical().copy( a );

			expect( a ).toEqualLikeQUnit( b );

			a.radius = 1;
			expect( a ).not.toEqualLikeQUnit( b );

		} );

		test( 'setFromVector3', () => {

			const a = new Cylindrical( 1, 1, 1 );
			const b = new Vector3( 0, 0, 0 );
			const c = new Vector3( 3, - 1, - 3 );
			const expected = new Cylindrical( Math.sqrt( 9 + 9 ), Math.atan2( 3, - 3 ), - 1 );

			a.setFromVector3( b );
			expect( a.radius ).toBe( 0 );
			expect( a.theta ).toBe( 0 );
			expect( a.y ).toBe( 0 );

			a.setFromVector3( c );
			expect( Math.abs( a.radius - expected.radius ) <= eps ).toBeTruthy();
			expect( Math.abs( a.theta - expected.theta ) <= eps ).toBeTruthy();
			expect( Math.abs( a.y - expected.y ) <= eps ).toBeTruthy();

		} );

	} );

} );

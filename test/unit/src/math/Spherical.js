import { describe, test, expect } from 'vitest';
import { Spherical } from '@src/math/Spherical.js';
import { Vector3 } from '@src/math/Vector3.js';
import {
	eps
} from '@test-utils/math-constants.js';

describe( 'Maths', () => {

	describe( 'Spherical', () => {

		test( 'Instancing', () => {

			let a = new Spherical();
			const radius = 10.0;
			const phi = Math.acos( - 0.5 );
			const theta = Math.sqrt( Math.PI ) * phi;

			expect( a.radius ).toBe( 1.0 );
			expect( a.phi ).toBe( 0 );
			expect( a.theta ).toBe( 0 );

			a = new Spherical( radius, phi, theta );
			expect( a.radius ).toBe( radius );
			expect( a.phi ).toBe( phi );
			expect( a.theta ).toBe( theta );

		} );

		test( 'set', () => {

			const a = new Spherical();
			const radius = 10.0;
			const phi = Math.acos( - 0.5 );
			const theta = Math.sqrt( Math.PI ) * phi;

			a.set( radius, phi, theta );
			expect( a.radius ).toBe( radius );
			expect( a.phi ).toBe( phi );
			expect( a.theta ).toBe( theta );

		} );

		test( 'clone', () => {

			const radius = 10.0;
			const phi = Math.acos( - 0.5 );
			const theta = Math.sqrt( Math.PI ) * phi;
			const a = new Spherical( radius, phi, theta );
			const b = a.clone();

			expect( a ).toEqualLikeQUnit( b );

			a.radius = 2.0;
			expect( a ).not.toEqualLikeQUnit( b );

		} );

		test( 'copy', () => {

			const radius = 10.0;
			const phi = Math.acos( - 0.5 );
			const theta = Math.sqrt( Math.PI ) * phi;
			const a = new Spherical( radius, phi, theta );
			const b = new Spherical().copy( a );

			expect( a ).toEqualLikeQUnit( b );

			a.radius = 2.0;
			expect( a ).not.toEqualLikeQUnit( b );

		} );

		test( 'makeSafe', () => {

			const EPS = 0.000001; // from source
			const tooLow = 0.0;
			const tooHigh = Math.PI;
			const justRight = 1.5;
			const a = new Spherical( 1, tooLow, 0 );

			a.makeSafe();
			expect( a.phi ).toBe( EPS );

			a.set( 1, tooHigh, 0 );
			a.makeSafe();
			expect( a.phi ).toBe( Math.PI - EPS );

			a.set( 1, justRight, 0 );
			a.makeSafe();
			expect( a.phi ).toBe( justRight );

		} );

		test( 'setFromVector3', () => {

			const a = new Spherical( 1, 1, 1 );
			const b = new Vector3( 0, 0, 0 );
			const c = new Vector3( Math.PI, 1, - Math.PI );
			const expected = new Spherical( 4.554032147688322, 1.3494066171539107, 2.356194490192345 );

			a.setFromVector3( b );
			expect( a.radius ).toBe( 0 );
			expect( a.phi ).toBe( 0 );
			expect( a.theta ).toBe( 0 );

			a.setFromVector3( c );
			expect( Math.abs( a.radius - expected.radius ) <= eps ).toBeTruthy();
			expect( Math.abs( a.phi - expected.phi ) <= eps ).toBeTruthy();
			expect( Math.abs( a.theta - expected.theta ) <= eps ).toBeTruthy();

		} );

		test( 'setFromCartesianCoords', () => {

			const a = new Spherical( 1, 1, 1 );
			const expected = new Spherical( 4.554032147688322, 1.3494066171539107, 2.356194490192345 );

			a.setFromCartesianCoords( 0, 0, 0 );
			expect( a.radius ).toBe( 0 );
			expect( a.phi ).toBe( 0 );
			expect( a.theta ).toBe( 0 );

			a.setFromCartesianCoords( Math.PI, 1, - Math.PI );
			expect( Math.abs( a.radius - expected.radius ) <= eps ).toBeTruthy();
			expect( Math.abs( a.phi - expected.phi ) <= eps ).toBeTruthy();
			expect( Math.abs( a.theta - expected.theta ) <= eps ).toBeTruthy();

		} );

	} );

} );

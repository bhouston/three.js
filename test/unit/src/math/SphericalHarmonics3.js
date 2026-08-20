import { describe, test, expect } from 'vitest';
import { SphericalHarmonics3 } from '@src/math/SphericalHarmonics3.js';

describe( 'Maths', () => {

	describe( 'SphericalHarmonics3', () => {

		test( 'Instancing', () => {

			const object = new SphericalHarmonics3();
			expect( object ).toBeTruthy();

		} );

		test( 'isSphericalHarmonics3', () => {

			const object = new SphericalHarmonics3();
			expect( object.isSphericalHarmonics3 ).toBeTruthy();

		} );

	} );

} );

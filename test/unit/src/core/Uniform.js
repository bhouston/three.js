import { describe, test, expect } from 'vitest';

import { Uniform } from '@src/core/Uniform.js';
import { Vector3 } from '@src/math/Vector3.js';
import {
	x,
	y,
	z
} from '@test-utils/math-constants.js';

describe( 'Core', () => {

	describe( 'Uniform', () => {

		test( 'Instancing', () => {

			let a;
			const b = new Vector3( x, y, z );

			a = new Uniform( 5 );
			expect( a.value ).toBe( 5 );

			a = new Uniform( b );
			expect( a.value.equals( b ) ).toBeTruthy();

		} );

		test( 'clone', () => {

			let a = new Uniform( 23 );
			let b = a.clone();

			expect( b.value ).toBe( a.value );

			a = new Uniform( new Vector3( 1, 2, 3 ) );
			b = a.clone();

			expect( b.value.equals( a.value ) ).toBeTruthy();

		} );

	} );

} );

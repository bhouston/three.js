import { describe, test, expect } from 'vitest';
import { CubicInterpolant } from '@src/math/interpolants/CubicInterpolant.js';
import { Interpolant } from '@src/math/Interpolant.js';

describe( 'Maths', () => {

	describe( 'Interpolants', () => {

		describe( 'CubicInterpolant', () => {

			test( 'Extending', () => {

				const object = new CubicInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object instanceof Interpolant ).toBe( true );

			} );

			test( 'Instancing', () => {

				// parameterPositions, sampleValues, sampleSize, resultBuffer
				const object = new CubicInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object ).toBeTruthy();

			} );

		} );

	} );

} );

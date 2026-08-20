import { describe, test, expect } from 'vitest';
import { QuaternionLinearInterpolant } from '@src/math/interpolants/QuaternionLinearInterpolant.js';
import { Interpolant } from '@src/math/Interpolant.js';

describe( 'Maths', () => {

	describe( 'Interpolants', () => {

		describe( 'QuaternionLinearInterpolant', () => {

			test( 'Extending', () => {

				const object = new QuaternionLinearInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object instanceof Interpolant ).toBe( true );

			} );

			test( 'Instancing', () => {

				// parameterPositions, sampleValues, sampleSize, resultBuffer
				const object = new QuaternionLinearInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object ).toBeTruthy();

			} );

		} );

	} );

} );

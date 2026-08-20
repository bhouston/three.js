import { describe, test, expect } from 'vitest';
import { LinearInterpolant } from '@src/math/interpolants/LinearInterpolant.js';
import { Interpolant } from '@src/math/Interpolant.js';

describe( 'Maths', () => {

	describe( 'Interpolants', () => {

		describe( 'LinearInterpolant', () => {

			test( 'Extending', () => {

				const object = new LinearInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object instanceof Interpolant ).toBe( true );

			} );

			test( 'Instancing', () => {

				// parameterPositions, sampleValues, sampleSize, resultBuffer
				const object = new LinearInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object ).toBeTruthy();

			} );

		} );

	} );

} );

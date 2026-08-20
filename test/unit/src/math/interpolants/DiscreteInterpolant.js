import { describe, test, expect } from 'vitest';
import { DiscreteInterpolant } from '@src/math/interpolants/DiscreteInterpolant.js';
import { Interpolant } from '@src/math/Interpolant.js';

describe( 'Maths', () => {

	describe( 'Interpolants', () => {

		describe( 'DiscreteInterpolant', () => {

			test( 'Extending', () => {

				const object = new DiscreteInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object instanceof Interpolant ).toBe( true );

			} );

			test( 'Instancing', () => {

				// parameterPositions, sampleValues, sampleSize, resultBuffer
				const object = new DiscreteInterpolant( null, [ 1, 11, 2, 22, 3, 33 ], 2, [] );
				expect( object ).toBeTruthy();

			} );

		} );

	} );

} );

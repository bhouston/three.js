import { describe, test, expect } from 'vitest';
import { CurvePath } from '@src/extras/core/CurvePath.js';

import { Curve } from '@src/extras/core/Curve.js';

describe( 'Extras', () => {

	describe( 'Core', () => {

		describe( 'CurvePath', () => {

			test( 'Extending', () => {

				const object = new CurvePath();
				expect( object instanceof Curve ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new CurvePath();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new Curve();
				expect( object.type === 'Curve' ).toBeTruthy();

			} );

		} );

	} );

} );

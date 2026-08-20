import { describe, test, expect } from 'vitest';
import { Path } from '@src/extras/core/Path.js';

import { CurvePath } from '@src/extras/core/CurvePath.js';

describe( 'Extras', () => {

	describe( 'Core', () => {

		describe( 'Path', () => {

			test( 'Extending', () => {

				const object = new Path();
				expect( object instanceof CurvePath ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new Path();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new Path();
				expect( object.type === 'Path' ).toBeTruthy();

			} );

		} );

	} );

} );

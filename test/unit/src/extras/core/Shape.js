import { describe, test, expect } from 'vitest';
import { Shape } from '@src/extras/core/Shape.js';

import { Path } from '@src/extras/core/Path.js';

describe( 'Extras', () => {

	describe( 'Core', () => {

		describe( 'Shape', () => {

			test( 'Extending', () => {

				const object = new Shape();
				expect( object instanceof Path ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new Shape();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new Shape();
				expect( object.type === 'Shape' ).toBeTruthy();

			} );

		} );

	} );

} );

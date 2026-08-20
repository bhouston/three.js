import { describe, test, expect } from 'vitest';

import { GridHelper } from '@src/helpers/GridHelper.js';

import { LineSegments } from '@src/objects/LineSegments.js';

describe( 'Helpers', () => {

	describe( 'GridHelper', () => {

		test( 'Extending', () => {

			const object = new GridHelper();
			expect( object instanceof LineSegments ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new GridHelper();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new GridHelper();
			expect( object.type === 'GridHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new GridHelper();
			object.dispose();

		} );

	} );

} );

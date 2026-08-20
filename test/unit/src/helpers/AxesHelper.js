import { describe, test, expect } from 'vitest';

import { AxesHelper } from '@src/helpers/AxesHelper.js';

import { LineSegments } from '@src/objects/LineSegments.js';

describe( 'Helpers', () => {

	describe( 'AxesHelper', () => {

		test( 'Extending', () => {

			const object = new AxesHelper();
			expect( object instanceof LineSegments ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new AxesHelper();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new AxesHelper();
			expect( object.type === 'AxesHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new AxesHelper();
			object.dispose();

		} );

	} );

} );

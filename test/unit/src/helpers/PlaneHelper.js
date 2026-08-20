import { describe, test, expect } from 'vitest';

import { PlaneHelper } from '@src/helpers/PlaneHelper.js';

import { Line } from '@src/objects/Line.js';

describe( 'Helpers', () => {

	describe( 'PlaneHelper', () => {

		test( 'Extending', () => {

			const object = new PlaneHelper();
			expect( object instanceof Line ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PlaneHelper();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new PlaneHelper();
			expect( object.type === 'PlaneHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new PlaneHelper();
			object.dispose();

		} );

	} );

} );

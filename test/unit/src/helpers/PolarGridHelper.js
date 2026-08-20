import { describe, test, expect } from 'vitest';

import { PolarGridHelper } from '@src/helpers/PolarGridHelper.js';

import { LineSegments } from '@src/objects/LineSegments.js';

describe( 'Helpers', () => {

	describe( 'PolarGridHelper', () => {

		test( 'Extending', () => {

			const object = new PolarGridHelper();
			expect( object instanceof LineSegments ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PolarGridHelper();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new PolarGridHelper();
			expect( object.type === 'PolarGridHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new PolarGridHelper();
			object.dispose();

		} );

	} );

} );

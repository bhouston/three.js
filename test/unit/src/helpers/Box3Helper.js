import { describe, test, expect } from 'vitest';

import { Box3Helper } from '@src/helpers/Box3Helper.js';

import { LineSegments } from '@src/objects/LineSegments.js';

describe( 'Helpers', () => {

	describe( 'Box3Helper', () => {

		test( 'Extending', () => {

			const object = new Box3Helper();
			expect( object instanceof LineSegments ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Box3Helper();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Box3Helper();
			expect( object.type === 'Box3Helper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new Box3Helper();
			object.dispose();

		} );

	} );

} );

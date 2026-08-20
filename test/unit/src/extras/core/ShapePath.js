import { describe, test, expect } from 'vitest';
import { ShapePath } from '@src/extras/core/ShapePath.js';

describe( 'Extras', () => {

	describe( 'Core', () => {

		describe( 'ShapePath', () => {

			test( 'Instancing', () => {

				const object = new ShapePath();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new ShapePath();
				expect( object.type === 'ShapePath' ).toBeTruthy();

			} );

		} );

	} );

} );

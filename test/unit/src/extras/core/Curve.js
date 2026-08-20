import { describe, test, expect } from 'vitest';
import { Curve } from '@src/extras/core/Curve.js';

describe( 'Extras', () => {

	describe( 'Core', () => {

		describe( 'Curve', () => {

			test( 'Instancing', () => {

				const object = new Curve();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new Curve();
				expect( object.type === 'Curve' ).toBeTruthy();

			} );

		} );

	} );

} );

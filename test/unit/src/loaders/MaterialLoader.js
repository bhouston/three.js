import { describe, test, expect } from 'vitest';

import { MaterialLoader } from '@src/loaders/MaterialLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'MaterialLoader', () => {

		test( 'Extending', () => {

			const object = new MaterialLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'textures', () => {

			const actual = new MaterialLoader().textures;
			const expected = {};
			expect( actual ).toEqual( expected );

		} );

		test( 'Instancing', () => {

			const object = new MaterialLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

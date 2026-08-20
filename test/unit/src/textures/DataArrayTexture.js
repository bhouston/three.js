import { describe, test, expect } from 'vitest';
import { DataArrayTexture } from '@src/textures/DataArrayTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'DataArrayTexture', () => {

		test( 'Extending', () => {

			const object = new DataArrayTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new DataArrayTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isDataArrayTexture', () => {

			const object = new DataArrayTexture();
			expect( object.isDataArrayTexture ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { DataTexture } from '@src/textures/DataTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'DataTexture', () => {

		test( 'Extending', () => {

			const object = new DataTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new DataTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isDataTexture', () => {

			const object = new DataTexture();
			expect( object.isDataTexture ).toBeTruthy();

		} );

	} );

} );

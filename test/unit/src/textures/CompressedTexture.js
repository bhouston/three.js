import { describe, test, expect } from 'vitest';
import { CompressedTexture } from '@src/textures/CompressedTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'CompressedTexture', () => {

		test( 'Extending', () => {

			const object = new CompressedTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CompressedTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isCompressedTexture', () => {

			const object = new CompressedTexture();
			expect( object.isCompressedTexture ).toBeTruthy();

		} );

	} );

} );

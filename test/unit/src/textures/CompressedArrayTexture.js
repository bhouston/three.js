import { describe, test, expect } from 'vitest';
import { CompressedArrayTexture } from '@src/textures/CompressedArrayTexture.js';
import { CompressedTexture } from '@src/textures/CompressedTexture.js';

describe( 'Textures', () => {

	describe( 'CompressedArrayTexture', () => {

		test( 'Extending', () => {

			const object = new CompressedArrayTexture();
			expect( object instanceof CompressedTexture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CompressedArrayTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isCompressedArrayTexture', () => {

			const object = new CompressedArrayTexture();
			expect( object.isCompressedArrayTexture ).toBeTruthy();

		} );

	} );

} );

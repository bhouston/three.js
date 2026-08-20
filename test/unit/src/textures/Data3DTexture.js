import { describe, test, expect } from 'vitest';
import { Data3DTexture } from '@src/textures/Data3DTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'Data3DTexture', () => {

		test( 'Extending', () => {

			const object = new Data3DTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Data3DTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isData3DTexture', () => {

			const object = new Data3DTexture();
			expect( object.isData3DTexture ).toBeTruthy();

		} );

	} );

} );

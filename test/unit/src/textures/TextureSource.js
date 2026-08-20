import { describe, test, expect } from 'vitest';
import { TextureSource } from '@src/textures/TextureSource.js';

describe( 'Textures', () => {

	describe( 'TextureSource', () => {

		test( 'Instancing', () => {

			const object = new TextureSource();
			expect( object ).toBeTruthy();

		} );

		test( 'isTextureSource', () => {

			const object = new TextureSource();
			expect( object.isTextureSource ).toBeTruthy();

		} );

	} );

} );

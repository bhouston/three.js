import { describe, test, expect } from 'vitest';
import { CubeTexture } from '@src/textures/CubeTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'CubeTexture', () => {

		test( 'Extending', () => {

			const object = new CubeTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CubeTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isCubeTexture', () => {

			const object = new CubeTexture();
			expect( object.isCubeTexture ).toBeTruthy();

		} );

	} );

} );

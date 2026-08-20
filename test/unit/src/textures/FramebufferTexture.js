import { describe, test, expect } from 'vitest';
import { FramebufferTexture } from '@src/textures/FramebufferTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'FramebufferTexture', () => {

		test( 'Extending', () => {

			const object = new FramebufferTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new FramebufferTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isFramebufferTexture', () => {

			const object = new FramebufferTexture();
			expect( object.isFramebufferTexture ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { Texture } from '@src/textures/Texture.js';
import { EventDispatcher } from '@src/core/EventDispatcher.js';

describe( 'Textures', () => {

	describe( 'Texture', () => {

		test( 'Extending', () => {

			const object = new Texture();
			expect( object instanceof EventDispatcher ).toBe( true );

		} );

		test( 'Instancing', () => {

			// no params
			const object = new Texture();
			expect( object ).toBeTruthy();

		} );

		test( 'isTexture', () => {

			const object = new Texture();
			expect( object.isTexture ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new Texture();
			object.dispose();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { DepthTexture } from '@src/textures/DepthTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'DepthTexture', () => {

		test( 'Extending', () => {

			const object = new DepthTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new DepthTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isDepthTexture', () => {

			const object = new DepthTexture();
			expect( object.isDepthTexture ).toBeTruthy();

		} );

	} );

} );

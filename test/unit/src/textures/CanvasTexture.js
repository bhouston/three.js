import { describe, test, expect } from 'vitest';
import { CanvasTexture } from '@src/textures/CanvasTexture.js';
import { Texture } from '@src/textures/Texture.js';

describe( 'Textures', () => {

	describe( 'CanvasTexture', () => {

		test( 'Extending', () => {

			const object = new CanvasTexture();
			expect( object instanceof Texture ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CanvasTexture();
			expect( object ).toBeTruthy();

		} );

		test( 'isCanvasTexture', () => {

			const object = new CanvasTexture();
			expect( object.isCanvasTexture ).toBeTruthy();

		} );

	} );

} );

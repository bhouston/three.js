import { describe, test, expect } from 'vitest';

import { ImageBitmapLoader } from '@src/loaders/ImageBitmapLoader.js';
import { Cache } from '@src/loaders/Cache.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'ImageBitmapLoader', () => {

		test( 'Extending', () => {

			const object = new ImageBitmapLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ImageBitmapLoader();
			expect( object ).toBeTruthy();

		} );

		test( 'options', () => {

			const actual = new ImageBitmapLoader().options;
			const expected = { premultiplyAlpha: 'none' };
			expect( actual ).toEqual( expected );

		} );

		test( 'isImageBitmapLoader', () => {

			const object = new ImageBitmapLoader();
			expect( object.isImageBitmapLoader ).toBeTruthy();

		} );

		test( 'load', async () => {

			const canvas = document.createElement( 'canvas' );
			canvas.width = 8;
			canvas.height = 8;

			const url = canvas.toDataURL( 'image/png' );

			const enabled = Cache.enabled;
			Cache.enabled = true;

			try {

				// concurrent requests for the same URL share the cached promise

				const [ first, second ] = await Promise.all( [
					new ImageBitmapLoader().loadAsync( url ),
					new ImageBitmapLoader().loadAsync( url )
				] );

				expect( first instanceof ImageBitmap ).toBeTruthy();
				expect( second instanceof ImageBitmap ).toBeTruthy();

			} finally {

				Cache.remove( `image-bitmap:${ url }` );
				Cache.enabled = enabled;

			}

		} );

	} );

} );

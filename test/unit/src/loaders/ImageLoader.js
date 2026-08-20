import { describe, test, expect } from 'vitest';

import { ImageLoader } from '@src/loaders/ImageLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'ImageLoader', () => {

		test( 'Extending', () => {

			const object = new ImageLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ImageLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

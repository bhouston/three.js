import { describe, test, expect } from 'vitest';

import { CompressedTextureLoader } from '@src/loaders/CompressedTextureLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'CompressedTextureLoader', () => {

		test( 'Extending', () => {

			const object = new CompressedTextureLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CompressedTextureLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

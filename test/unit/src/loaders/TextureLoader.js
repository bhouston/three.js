import { describe, test, expect } from 'vitest';

import { TextureLoader } from '@src/loaders/TextureLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'TextureLoader', () => {

		test( 'Extending', () => {

			const object = new TextureLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new TextureLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

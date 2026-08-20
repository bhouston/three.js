import { describe, test, expect } from 'vitest';

import { CubeTextureLoader } from '@src/loaders/CubeTextureLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'CubeTextureLoader', () => {

		test( 'Extending', () => {

			const object = new CubeTextureLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CubeTextureLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

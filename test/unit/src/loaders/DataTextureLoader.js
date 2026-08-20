import { describe, test, expect } from 'vitest';

import { DataTextureLoader } from '@src/loaders/DataTextureLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'DataTextureLoader', () => {

		test( 'Extending', () => {

			const object = new DataTextureLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new DataTextureLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

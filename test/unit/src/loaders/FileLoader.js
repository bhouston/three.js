import { describe, test, expect } from 'vitest';

import { FileLoader } from '@src/loaders/FileLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'FileLoader', () => {

		test( 'Extending', () => {

			const object = new FileLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new FileLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

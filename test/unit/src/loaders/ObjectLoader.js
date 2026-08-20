import { describe, test, expect } from 'vitest';

import { ObjectLoader } from '@src/loaders/ObjectLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'ObjectLoader', () => {

		test( 'Extending', () => {

			const object = new ObjectLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ObjectLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

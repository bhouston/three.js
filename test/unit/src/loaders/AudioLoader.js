import { describe, test, expect } from 'vitest';

import { AudioLoader } from '@src/loaders/AudioLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'AudioLoader', () => {

		test( 'Extending', () => {

			const object = new AudioLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new AudioLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

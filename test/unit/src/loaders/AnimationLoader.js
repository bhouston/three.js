import { describe, test, expect } from 'vitest';

import { AnimationLoader } from '@src/loaders/AnimationLoader.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'AnimationLoader', () => {

		test( 'Extending', () => {

			const object = new AnimationLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new AnimationLoader();
			expect( object ).toBeTruthy();

		} );

	} );

} );

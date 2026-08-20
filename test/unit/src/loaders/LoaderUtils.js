import { describe, test, expect } from 'vitest';

import { LoaderUtils } from '@src/loaders/LoaderUtils.js';

describe( 'Loaders', () => {

	describe( 'LoaderUtils', () => {

		test( 'extractUrlBase', () => {

			expect( LoaderUtils.extractUrlBase( '/path/to/model.glb' ) ).toBe( '/path/to/' );
			expect( LoaderUtils.extractUrlBase( 'model.glb' ) ).toBe( './' );
			expect( LoaderUtils.extractUrlBase( '/model.glb' ) ).toBe( '/' );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';

import { Cache } from '@src/loaders/Cache.js';

describe( 'Loaders', () => {

	describe( 'Cache', () => {

		test( 'enabled', () => {

			const actual = Cache.enabled;
			const expected = false;
			expect( actual ).toBe( expected );

		} );

		test( 'files', () => {

			const actual = Cache.files;
			const expected = {};
			expect( actual ).toEqual( expected );

		} );

	} );

} );

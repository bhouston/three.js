import { describe, test, expect } from 'vitest';

import { Loader } from '@src/loaders/Loader.js';
import { LoadingManager } from '@src/loaders/LoadingManager.js';

describe( 'Loaders', () => {

	describe( 'Loader', () => {

		test( 'Instancing', () => {

			const object = new Loader();
			expect( object ).toBeTruthy();

		} );

		test( 'manager', () => {

			// uses default LoadingManager if not supplied in constructor
			const object = new Loader().manager;
			expect( object instanceof LoadingManager ).toBe( true );

		} );

		test( 'crossOrigin', () => {

			const actual = new Loader().crossOrigin;
			const expected = 'anonymous';
			expect( actual ).toBe( expected );

		} );

		test( 'withCredentials', () => {

			const actual = new Loader().withCredentials;
			const expected = false;
			expect( actual ).toBe( expected );

		} );

		test( 'path', () => {

			const actual = new Loader().path;
			const expected = '';
			expect( actual ).toBe( expected );

		} );

		test( 'resourcePath', () => {

			const actual = new Loader().resourcePath;
			const expected = '';
			expect( actual ).toBe( expected );

		} );

		test( 'requestHeader', () => {

			const actual = new Loader().requestHeader;
			const expected = {};
			expect( actual ).toEqual( expected );

		} );

	} );

} );

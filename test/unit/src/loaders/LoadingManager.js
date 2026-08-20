import { describe, test, expect } from 'vitest';

import { LoadingManager } from '@src/loaders/LoadingManager.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'LoadingManager', () => {

		test( 'Instancing', () => {

			// no params
			const object = new LoadingManager();
			expect( object ).toBeTruthy();

			// onLoad, onProgress, onError

		} );

		test( 'addHandler/getHandler/removeHandler', () => {

			const loadingManager = new LoadingManager();
			const loader = new Loader();

			const regex1 = /\.jpg$/i;
			const regex2 = /\.jpg$/gi;

			loadingManager.addHandler( regex1, loader );

			expect( loadingManager.getHandler( 'foo.jpg' ) ).toBe( loader );
			expect( loadingManager.getHandler( 'foo.jpg.png' ) ).toBe( null );
			expect( loadingManager.getHandler( 'foo.jpeg' ) ).toBe( null );

			loadingManager.removeHandler( regex1 );
			loadingManager.addHandler( regex2, loader );

			expect( loadingManager.getHandler( 'foo.jpg' ) ).toBe( loader );
			// Test twice, see #17920.
			expect( loadingManager.getHandler( 'foo.jpg' ) ).toBe( loader );

		} );

		test( 'abortController - lazy instantiation', () => {

			const loadingManager = new LoadingManager();

			expect( loadingManager._abortController ).toBe( null );

			const controller = loadingManager.abortController;

			expect( controller instanceof AbortController ).toBeTruthy();
			expect( loadingManager._abortController ).toBe( controller );

			const controller2 = loadingManager.abortController;
			expect( controller ).toBe( controller2 );

		} );

		test( 'abort() - aborts controller and resets', () => {

			const loadingManager = new LoadingManager();

			const controller = loadingManager.abortController;

			expect( ! controller.signal.aborted ).toBeTruthy();

			loadingManager.abort();

			expect( controller.signal.aborted ).toBeTruthy();
			expect( loadingManager._abortController ).toBe( null );

		} );

		test( 'abortController - recreation after abort', () => {

			const loadingManager = new LoadingManager();

			const controller1 = loadingManager.abortController;

			loadingManager.abort();

			expect( controller1.signal.aborted ).toBeTruthy();
			expect( loadingManager._abortController ).toBe( null );

			const controller2 = loadingManager.abortController;

			expect( controller2 instanceof AbortController ).toBeTruthy();
			expect( controller1 ).not.toBe( controller2 );
			expect( ! controller2.signal.aborted ).toBeTruthy();

		} );

	} );

} );

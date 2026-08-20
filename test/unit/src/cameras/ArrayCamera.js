import { describe, test, expect } from 'vitest';
import { ArrayCamera } from '@src/cameras/ArrayCamera.js';
import { PerspectiveCamera } from '@src/cameras/PerspectiveCamera.js';

describe( 'Cameras', () => {

	describe( 'ArrayCamera', () => {

		test( 'Extending', () => {

			const object = new ArrayCamera();
			expect( object instanceof PerspectiveCamera ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ArrayCamera();
			expect( object ).toBeTruthy();

		} );

		test( 'isArrayCamera', () => {

			const object = new ArrayCamera();
			expect( object.isArrayCamera ).toBeTruthy();

		} );

	} );

} );

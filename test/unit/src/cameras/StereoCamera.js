import { describe, test, expect } from 'vitest';
import { StereoCamera } from '@src/cameras/StereoCamera.js';

describe( 'Cameras', () => {

	describe( 'StereoCamera', () => {

		test( 'Instancing', () => {

			const object = new StereoCamera();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new StereoCamera();
			expect( object.type === 'StereoCamera' ).toBeTruthy();

		} );

	} );

} );

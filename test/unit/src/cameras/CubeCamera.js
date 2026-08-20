import { describe, test, expect } from 'vitest';
import { CubeCamera } from '@src/cameras/CubeCamera.js';
import { Object3D } from '@src/core/Object3D.js';

describe( 'Cameras', () => {

	describe( 'CubeCamera', () => {

		test( 'Extending', () => {

			const object = new CubeCamera();
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CubeCamera();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new CubeCamera();
			expect( object.type === 'CubeCamera' ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { OrthographicCamera } from '@src/cameras/OrthographicCamera.js';
import { Camera } from '@src/cameras/Camera.js';

describe( 'Cameras', () => {

	describe( 'OrthographicCamera', () => {

		test( 'Extending', () => {

			const object = new OrthographicCamera();
			expect( object instanceof Camera ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new OrthographicCamera();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new OrthographicCamera();
			expect( object.type === 'OrthographicCamera' ).toBeTruthy();

		} );

		test( 'isOrthographicCamera', () => {

			const object = new OrthographicCamera();
			expect( object.isOrthographicCamera ).toBeTruthy();

		} );

		test( 'updateProjectionMatrix', () => {

			const left = - 1, right = 1, top = 1, bottom = - 1, near = 1, far = 3;
			const cam = new OrthographicCamera( left, right, top, bottom, near, far );

			// updateProjectionMatrix is called in constructor
			const pMatrix = cam.projectionMatrix.elements;

			// orthographic projection is given my the 4x4 Matrix
			// 2/r-l		0			 0		-(l+r/r-l)
			//   0		2/t-b		 0		-(t+b/t-b)
			//   0			0		-2/f-n	-(f+n/f-n)
			//   0			0			 0				1

			expect( pMatrix[ 0 ] === 2 / ( right - left ) ).toBeTruthy();
			expect( pMatrix[ 5 ] === 2 / ( top - bottom ) ).toBeTruthy();
			expect( pMatrix[ 10 ] === - 2 / ( far - near ) ).toBeTruthy();
			expect( pMatrix[ 12 ] === - ( ( right + left ) / ( right - left ) ) ).toBeTruthy();
			expect( pMatrix[ 13 ] === - ( ( top + bottom ) / ( top - bottom ) ) ).toBeTruthy();
			expect( pMatrix[ 14 ] === - ( ( far + near ) / ( far - near ) ) ).toBeTruthy();

		} );

		// TODO: clone is a camera methods that relied to copy method
		test( 'clone', () => {

			const left = - 1.5, right = 1.5, top = 1, bottom = - 1, near = 0.1, far = 42;
			const cam = new OrthographicCamera( left, right, top, bottom, near, far );

			const clonedCam = cam.clone();

			expect( cam.left === clonedCam.left ).toBeTruthy();
			expect( cam.right === clonedCam.right ).toBeTruthy();
			expect( cam.top === clonedCam.top ).toBeTruthy();
			expect( cam.bottom === clonedCam.bottom ).toBeTruthy();
			expect( cam.near === clonedCam.near ).toBeTruthy();
			expect( cam.far === clonedCam.far ).toBeTruthy();
			expect( cam.zoom === clonedCam.zoom ).toBeTruthy();

		} );

	} );

} );

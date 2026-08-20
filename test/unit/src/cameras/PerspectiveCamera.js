import { describe, test, expect } from 'vitest';
import { PerspectiveCamera } from '@src/cameras/PerspectiveCamera.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import { Camera } from '@src/cameras/Camera.js';

describe( 'Cameras', () => {

	describe( 'PerspectiveCamera', () => {

		// see e.g. math/Matrix4.js
		const matrixEquals4 = function ( a, b, tolerance ) {

			tolerance = tolerance || 0.0001;
			if ( a.elements.length != b.elements.length ) {

				return false;

			}

			for ( let i = 0, il = a.elements.length; i < il; i ++ ) {

				const delta = Math.abs( a.elements[ i ] - b.elements[ i ] );
				if ( delta > tolerance ) {

					return false;

				}

			}

			return true;

		};

		test( 'Extending', () => {

			const object = new PerspectiveCamera();
			expect( object instanceof Camera ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PerspectiveCamera();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new PerspectiveCamera();
			expect( object.type === 'PerspectiveCamera' ).toBeTruthy();

		} );

		test( 'isPerspectiveCamera', () => {

			const object = new PerspectiveCamera();
			expect( object.isPerspectiveCamera ).toBeTruthy();

		} );

		test( 'updateProjectionMatrix', () => {

			const cam = new PerspectiveCamera( 75, 16 / 9, 0.1, 300.0 );

			// updateProjectionMatrix is called in constructor
			const m = cam.projectionMatrix;

			// perspective projection is given my the 4x4 Matrix
			// 2n/r-l		0			l+r/r-l				 0
			//   0		2n/t-b	t+b/t-b				 0
			//   0			0		-(f+n/f-n)	-(2fn/f-n)
			//   0			0				-1					 0

			// this matrix was calculated by hand via glMatrix.perspective(75, 16 / 9, 0.1, 300.0, pMatrix)
			// to get a reference matrix from plain WebGL
			const reference = new Matrix4().set(
				0.7330642938613892, 0, 0, 0,
				0, 1.3032253980636597, 0, 0,
				0, 0, - 1.000666856765747, - 0.2000666856765747,
				0, 0, - 1, 0
			);

			expect( matrixEquals4( reference, m, 0.000001 ) ).toBeTruthy();

		} );

		// TODO: clone is a camera methods that relied to copy method
		test( 'clone', () => {

			const near = 1,
				far = 3,
				aspect = 16 / 9,
				fov = 90;

			const cam = new PerspectiveCamera( fov, aspect, near, far );

			const clonedCam = cam.clone();

			expect( cam.fov === clonedCam.fov ).toBeTruthy();
			expect( cam.aspect === clonedCam.aspect ).toBeTruthy();
			expect( cam.near === clonedCam.near ).toBeTruthy();
			expect( cam.far === clonedCam.far ).toBeTruthy();
			expect( cam.zoom === clonedCam.zoom ).toBeTruthy();
			expect( cam.projectionMatrix.equals( clonedCam.projectionMatrix ) ).toBeTruthy();

		} );

	} );

} );

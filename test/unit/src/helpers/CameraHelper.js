import { describe, test, expect } from 'vitest';

import { CameraHelper } from '@src/helpers/CameraHelper.js';

import { LineSegments } from '@src/objects/LineSegments.js';
import { PerspectiveCamera } from '@src/cameras/PerspectiveCamera.js';

describe( 'Helpers', () => {

	describe( 'CameraHelper', () => {

		test( 'Extending', () => {

			const camera = new PerspectiveCamera();
			const object = new CameraHelper( camera );
			expect( object instanceof LineSegments ).toBe( true );

		} );

		test( 'Instancing', () => {

			const camera = new PerspectiveCamera();
			const object = new CameraHelper( camera );
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const camera = new PerspectiveCamera();
			const object = new CameraHelper( camera );
			expect( object.type === 'CameraHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const camera = new PerspectiveCamera();
			const object = new CameraHelper( camera );
			object.dispose();

		} );

	} );

} );

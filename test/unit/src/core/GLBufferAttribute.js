import { describe, test, expect } from 'vitest';

import { GLBufferAttribute } from '@src/core/GLBufferAttribute.js';

describe( 'Core', () => {

	describe( 'GLBufferAttribute', () => {

		test( 'Instancing', () => {

			const object = new GLBufferAttribute();
			expect( object ).toBeTruthy();

		} );

		test( 'isGLBufferAttribute', () => {

			const object = new GLBufferAttribute();
			expect( object.isGLBufferAttribute ).toBeTruthy();

		} );

	} );

} );

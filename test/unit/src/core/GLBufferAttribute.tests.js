import { describe, test } from 'vitest';
import { assert } from '../../utils/assertions.js';

import { GLBufferAttribute } from '../../../../src/core/GLBufferAttribute.js';

describe( 'Core', () => {

	describe( 'GLBufferAttribute', () => {

		// INSTANCING
		test( 'Instancing', () => {

			const object = new GLBufferAttribute();
			assert.ok( object, 'Can instantiate a GLBufferAttribute.' );

		} );

		// PUBLIC
		test( 'isGLBufferAttribute', () => {

			const object = new GLBufferAttribute();
			assert.ok(
				object.isGLBufferAttribute,
				'GLBufferAttribute.isGLBufferAttribute should be true'
			);

		} );

	} );

} );

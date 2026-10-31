import { describe, test } from 'vitest';
import { assert } from '../../../utils/assertions.js';

import { UniformsLib } from '../../../../../src/renderers/shaders/UniformsLib.js';

describe( 'Renderers', () => {

	describe( 'Shaders', () => {

		describe( 'UniformsLib', () => {

			// INSTANCING
			test( 'Instancing', () => {

				assert.ok( UniformsLib, 'UniformsLib is defined.' );

			} );

		} );

	} );

} );

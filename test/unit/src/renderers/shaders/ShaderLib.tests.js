import { describe, test } from 'vitest';
import { assert } from '../../../utils/assertions.js';

import { ShaderLib } from '../../../../../src/renderers/shaders/ShaderLib.js';

describe( 'Renderers', () => {

	describe( 'Shaders', () => {

		describe( 'ShaderLib', () => {

			// INSTANCING
			test( 'Instancing', () => {

				assert.ok( ShaderLib, 'ShaderLib is defined.' );

			} );

		} );

	} );

} );

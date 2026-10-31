import { describe, test } from 'vitest';
import { assert } from '../../../utils/assertions.js';

import { ShaderChunk } from '../../../../../src/renderers/shaders/ShaderChunk.js';

describe( 'Renderers', () => {

	describe( 'Shaders', () => {

		describe( 'ShaderChunk', () => {

			// INSTANCING
			test( 'Instancing', () => {

				assert.ok( ShaderChunk, 'ShaderChunk is defined.' );

			} );

		} );

	} );

} );

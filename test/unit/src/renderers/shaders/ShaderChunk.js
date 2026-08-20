import { describe, test, expect } from 'vitest';
import { ShaderChunk } from '@src/renderers/shaders/ShaderChunk.js';

describe( 'Renderers', () => {

	describe( 'Shaders', () => {

		describe( 'ShaderChunk', () => {

			test( 'Instancing', () => {

				expect( ShaderChunk ).toBeTruthy();

			} );

		} );

	} );

} );

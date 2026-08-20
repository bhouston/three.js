import { describe, test, expect } from 'vitest';
import { ShaderLib } from '@src/renderers/shaders/ShaderLib.js';

describe( 'Renderers', () => {

	describe( 'Shaders', () => {

		describe( 'ShaderLib', () => {

			test( 'Instancing', () => {

				expect( ShaderLib ).toBeTruthy();

			} );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { UniformsLib } from '@src/renderers/shaders/UniformsLib.js';

describe( 'Renderers', () => {

	describe( 'Shaders', () => {

		describe( 'UniformsLib', () => {

			test( 'Instancing', () => {

				expect( UniformsLib ).toBeTruthy();

			} );

		} );

	} );

} );

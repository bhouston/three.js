import { describe, test, expect } from 'vitest';
import { NearestFilter } from '@src/constants.js';
import { WebGLCubeRenderTarget } from '@src/renderers/WebGLCubeRenderTarget.js';

import { WebGLRenderTarget } from '@src/renderers/WebGLRenderTarget.js';

describe( 'Renderers', () => {

	describe( 'WebGLCubeRenderTarget', () => {

		test( 'Extending', () => {

			const object = new WebGLCubeRenderTarget();
			expect( object instanceof WebGLRenderTarget ).toBe( true );

			const options = new WebGLCubeRenderTarget( 1, { magFilter: NearestFilter } );
			expect( options.width === 1 && options.height === 1 && options.texture.magFilter === NearestFilter ).toBeTruthy();

		} );

		test( 'Instancing', () => {

			const object = new WebGLCubeRenderTarget();
			expect( object ).toBeTruthy();

		} );

	} );

} );

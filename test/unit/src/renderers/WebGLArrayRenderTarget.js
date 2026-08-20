import { describe, test, expect } from 'vitest';
import { NearestFilter } from '@src/constants.js';
import { WebGLArrayRenderTarget } from '@src/renderers/WebGLArrayRenderTarget.js';

import { WebGLRenderTarget } from '@src/renderers/WebGLRenderTarget.js';

describe( 'Renderers', () => {

	describe( 'WebGLArrayRenderTarget', () => {

		test( 'Extending', () => {

			const object = new WebGLArrayRenderTarget();
			expect( object instanceof WebGLRenderTarget ).toBe( true );

			const options = new WebGLArrayRenderTarget( 1, 1, 1, { magFilter: NearestFilter } );
			expect( options.width === 1 && options.height === 1 && options.depth === 1 && options.texture.magFilter === NearestFilter ).toBeTruthy();

		} );

		test( 'Instancing', () => {

			const object = new WebGLArrayRenderTarget();
			expect( object ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { NearestFilter } from '@src/constants.js';
import { WebGL3DRenderTarget } from '@src/renderers/WebGL3DRenderTarget.js';

import { WebGLRenderTarget } from '@src/renderers/WebGLRenderTarget.js';

describe( 'Renderers', () => {

	describe( 'WebGL3DRenderTarget', () => {

		test( 'Extending', () => {

			const object = new WebGL3DRenderTarget();
			expect( object instanceof WebGLRenderTarget ).toBe( true );

			const options = new WebGL3DRenderTarget( 1, 1, 1, { magFilter: NearestFilter } );
			expect( options.width === 1 && options.height === 1 && options.depth === 1 && options.texture.magFilter === NearestFilter ).toBeTruthy();

		} );

		test( 'Instancing', () => {

			const object = new WebGL3DRenderTarget();
			expect( object ).toBeTruthy();

		} );

	} );

} );

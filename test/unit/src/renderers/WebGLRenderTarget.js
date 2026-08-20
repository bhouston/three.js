import { describe, test, expect } from 'vitest';
import { WebGLRenderTarget } from '@src/renderers/WebGLRenderTarget.js';

import { EventDispatcher } from '@src/core/EventDispatcher.js';
import { NearestFilter } from '@src/constants.js';

describe( 'Renderers', () => {

	describe( 'WebGLRenderTarget', () => {

		test( 'Extending', () => {

			const object = new WebGLRenderTarget();
			expect( object instanceof EventDispatcher ).toBe( true );

			const options = new WebGLRenderTarget( 1, 1, { magFilter: NearestFilter } );
			expect( options.width === 1 && options.height === 1 && options.texture.magFilter === NearestFilter ).toBeTruthy();

		} );

		test( 'Instancing', () => {

			const object = new WebGLRenderTarget();
			expect( object ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new WebGLRenderTarget();
			object.dispose();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';

import { RenderTarget3D } from '@src/core/RenderTarget3D.js';
import { EventDispatcher } from '@src/core/EventDispatcher.js';
import { NearestFilter, RepeatWrapping } from '@src/constants.js';

describe( 'Core', () => {

	describe( 'RenderTarget3D', () => {

		test( 'Extending', () => {

			const object = new RenderTarget3D();
			expect( object instanceof EventDispatcher ).toBe( true );

			const options = new RenderTarget3D( 1, 1, 1, { magFilter: NearestFilter, wrapR: RepeatWrapping } );
			expect( options.width === 1 && options.height === 1 && options.depth === 1 && options.texture.magFilter === NearestFilter && options.texture.wrapR === RepeatWrapping ).toBeTruthy();

		} );

	} );

} );

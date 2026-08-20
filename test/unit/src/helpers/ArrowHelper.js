import { describe, test, expect } from 'vitest';

import { ArrowHelper } from '@src/helpers/ArrowHelper.js';

import { Object3D } from '@src/core/Object3D.js';

describe( 'Helpers', () => {

	describe( 'ArrowHelper', () => {

		test( 'Extending', () => {

			const object = new ArrowHelper();
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ArrowHelper();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new ArrowHelper();
			expect( object.type === 'ArrowHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new ArrowHelper();
			object.dispose();

		} );

	} );

} );

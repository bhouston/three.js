import { describe, test, expect } from 'vitest';
import { Fog } from '@src/scenes/Fog.js';

describe( 'Scenes', () => {

	describe( 'Fog', () => {

		test( 'Instancing', () => {

			// Fog( color, near = 1, far = 1000 )

			// no params
			const object = new Fog();
			expect( object ).toBeTruthy();

			// color
			const object_color = new Fog( 0xffffff );
			expect( object_color ).toBeTruthy();

			// color, near, far
			const object_all = new Fog( 0xffffff, 0.015, 100 );
			expect( object_all ).toBeTruthy();

		} );

		test( 'isFog', () => {

			const object = new Fog();
			expect( object.isFog ).toBeTruthy();

		} );

	} );

} );

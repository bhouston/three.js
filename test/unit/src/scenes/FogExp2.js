import { describe, test, expect } from 'vitest';
import { FogExp2 } from '@src/scenes/FogExp2.js';

describe( 'Scenes', () => {

	describe( 'FoxExp2', () => {

		test( 'Instancing', () => {

			// FoxExp2( color, density = 0.00025 )

			// no params
			const object = new FogExp2();
			expect( object ).toBeTruthy();

			// color
			const object_color = new FogExp2( 0xffffff );
			expect( object_color ).toBeTruthy();

			// color, density
			const object_all = new FogExp2( 0xffffff, 0.00030 );
			expect( object_all ).toBeTruthy();

		} );

		test( 'isFogExp2', () => {

			const object = new FogExp2();
			expect( object.isFogExp2 ).toBeTruthy();

		} );

	} );

} );

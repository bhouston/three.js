import { describe, test, expect } from 'vitest';
import { PointLightShadow } from '@src/lights/PointLightShadow.js';
import { LightShadow } from '@src/lights/LightShadow.js';

describe( 'Lights', () => {

	describe( 'PointLightShadow', () => {

		test( 'Extending', () => {

			const object = new PointLightShadow();
			expect( object instanceof LightShadow ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PointLightShadow();
			expect( object ).toBeTruthy();

		} );

		test( 'isPointLightShadow', () => {

			const object = new PointLightShadow();
			expect( object.isPointLightShadow ).toBeTruthy();

		} );

	} );

} );

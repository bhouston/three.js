import { describe, test, expect } from 'vitest';
import { ShadowMaterial } from '@src/materials/ShadowMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'ShadowMaterial', () => {

		test( 'Extending', () => {

			const object = new ShadowMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ShadowMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new ShadowMaterial();
			expect( object.type === 'ShadowMaterial' ).toBeTruthy();

		} );

		test( 'isShadowMaterial', () => {

			const object = new ShadowMaterial();
			expect( object.isShadowMaterial ).toBeTruthy();

		} );

	} );

} );

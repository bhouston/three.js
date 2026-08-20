import { describe, test, expect } from 'vitest';
import { ShaderMaterial } from '@src/materials/ShaderMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'ShaderMaterial', () => {

		test( 'Extending', () => {

			const object = new ShaderMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ShaderMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new ShaderMaterial();
			expect( object.type === 'ShaderMaterial' ).toBeTruthy();

		} );

		test( 'isShaderMaterial', () => {

			const object = new ShaderMaterial();
			expect( object.isShaderMaterial ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { RawShaderMaterial } from '@src/materials/RawShaderMaterial.js';
import { ShaderMaterial } from '@src/materials/ShaderMaterial.js';

describe( 'Materials', () => {

	describe( 'RawShaderMaterial', () => {

		test( 'Extending', () => {

			const object = new RawShaderMaterial();
			expect( object instanceof ShaderMaterial ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new RawShaderMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new RawShaderMaterial();
			expect( object.type === 'RawShaderMaterial' ).toBeTruthy();

		} );

		test( 'isRawShaderMaterial', () => {

			const object = new RawShaderMaterial();
			expect( object.isRawShaderMaterial ).toBeTruthy();

		} );

	} );

} );

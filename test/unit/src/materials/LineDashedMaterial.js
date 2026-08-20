import { describe, test, expect } from 'vitest';
import { LineDashedMaterial } from '@src/materials/LineDashedMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'LineDashedMaterial', () => {

		test( 'Extending', () => {

			const object = new LineDashedMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new LineDashedMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new LineDashedMaterial();
			expect( object.type === 'LineDashedMaterial' ).toBeTruthy();

		} );

		test( 'isLineDashedMaterial', () => {

			const object = new LineDashedMaterial();
			expect( object.isLineDashedMaterial ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { LineBasicMaterial } from '@src/materials/LineBasicMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'LineBasicMaterial', () => {

		test( 'Extending', () => {

			const object = new LineBasicMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new LineBasicMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new LineBasicMaterial();
			expect( object.type === 'LineBasicMaterial' ).toBeTruthy();

		} );

		test( 'isLineBasicMaterial', () => {

			const object = new LineBasicMaterial();
			expect( object.isLineBasicMaterial ).toBeTruthy();

		} );

	} );

} );

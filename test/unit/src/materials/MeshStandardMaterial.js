import { describe, test, expect } from 'vitest';
import { MeshStandardMaterial } from '@src/materials/MeshStandardMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshStandardMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshStandardMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshStandardMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshStandardMaterial();
			expect( object.type === 'MeshStandardMaterial' ).toBeTruthy();

		} );

		test( 'isMeshStandardMaterial', () => {

			const object = new MeshStandardMaterial();
			expect( object.isMeshStandardMaterial ).toBeTruthy();

		} );

	} );

} );

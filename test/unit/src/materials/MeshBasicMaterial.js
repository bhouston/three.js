import { describe, test, expect } from 'vitest';
import { MeshBasicMaterial } from '@src/materials/MeshBasicMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshBasicMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshBasicMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshBasicMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshBasicMaterial();
			expect( object.type === 'MeshBasicMaterial' ).toBeTruthy();

		} );

		test( 'isMeshBasicMaterial', () => {

			const object = new MeshBasicMaterial();
			expect( object.isMeshBasicMaterial ).toBeTruthy();

		} );

	} );

} );

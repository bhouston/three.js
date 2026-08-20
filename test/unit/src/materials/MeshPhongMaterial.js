import { describe, test, expect } from 'vitest';
import { MeshPhongMaterial } from '@src/materials/MeshPhongMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshPhongMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshPhongMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshPhongMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshPhongMaterial();
			expect( object.type === 'MeshPhongMaterial' ).toBeTruthy();

		} );

		test( 'isMeshPhongMaterial', () => {

			const object = new MeshPhongMaterial();
			expect( object.isMeshPhongMaterial ).toBeTruthy();

		} );

	} );

} );

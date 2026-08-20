import { describe, test, expect } from 'vitest';
import { MeshDepthMaterial } from '@src/materials/MeshDepthMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshDepthMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshDepthMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshDepthMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshDepthMaterial();
			expect( object.type === 'MeshDepthMaterial' ).toBeTruthy();

		} );

		test( 'isMeshDepthMaterial', () => {

			const object = new MeshDepthMaterial();
			expect( object.isMeshDepthMaterial ).toBeTruthy();

		} );

	} );

} );

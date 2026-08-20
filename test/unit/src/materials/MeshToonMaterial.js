import { describe, test, expect } from 'vitest';
import { MeshToonMaterial } from '@src/materials/MeshToonMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshToonMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshToonMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshToonMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshToonMaterial();
			expect( object.type === 'MeshToonMaterial' ).toBeTruthy();

		} );

		test( 'isMeshToonMaterial', () => {

			const object = new MeshToonMaterial();
			expect( object.isMeshToonMaterial ).toBeTruthy();

		} );

	} );

} );

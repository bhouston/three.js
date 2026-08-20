import { describe, test, expect } from 'vitest';
import { MeshNormalMaterial } from '@src/materials/MeshNormalMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshNormalMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshNormalMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshNormalMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshNormalMaterial();
			expect( object.type === 'MeshNormalMaterial' ).toBeTruthy();

		} );

		test( 'isMeshNormalMaterial', () => {

			const object = new MeshNormalMaterial();
			expect( object.isMeshNormalMaterial ).toBeTruthy();

		} );

	} );

} );

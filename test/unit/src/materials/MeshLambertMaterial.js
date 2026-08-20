import { describe, test, expect } from 'vitest';
import { MeshLambertMaterial } from '@src/materials/MeshLambertMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshLambertMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshLambertMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshLambertMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshLambertMaterial();
			expect( object.type === 'MeshLambertMaterial' ).toBeTruthy();

		} );

		test( 'isMeshLambertMaterial', () => {

			const object = new MeshLambertMaterial();
			expect( object.isMeshLambertMaterial ).toBeTruthy();

		} );

	} );

} );

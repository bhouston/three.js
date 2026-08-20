import { describe, test, expect } from 'vitest';
import { MeshDistanceMaterial } from '@src/materials/MeshDistanceMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshDistanceMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshDistanceMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshDistanceMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshDistanceMaterial();
			expect( object.type === 'MeshDistanceMaterial' ).toBeTruthy();

		} );

		test( 'isMeshDistanceMaterial', () => {

			const object = new MeshDistanceMaterial();
			expect( object.isMeshDistanceMaterial ).toBeTruthy();

		} );

	} );

} );

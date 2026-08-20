import { describe, test, expect } from 'vitest';
import { MeshMatcapMaterial } from '@src/materials/MeshMatcapMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshMatcapMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshMatcapMaterial();

			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshMatcapMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'defines', () => {

			const actual = new MeshMatcapMaterial().defines;
			const expected = { 'MATCAP': '' };
			expect( actual ).toEqual( expected );

		} );

		test( 'type', () => {

			const object = new MeshMatcapMaterial();
			expect( object.type === 'MeshMatcapMaterial' ).toBeTruthy();

		} );

		test( 'isMeshMatcapMaterial', () => {

			const object = new MeshMatcapMaterial();
			expect( object.isMeshMatcapMaterial ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { MeshPhysicalMaterial } from '@src/materials/MeshPhysicalMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'MeshPhysicalMaterial', () => {

		test( 'Extending', () => {

			const object = new MeshPhysicalMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new MeshPhysicalMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new MeshPhysicalMaterial();
			expect( object.type === 'MeshPhysicalMaterial' ).toBeTruthy();

		} );

		test( 'retroreflectivity', () => {

			const object = new MeshPhysicalMaterial();
			expect( object.retroreflectivity ).toBe( 0 );

			object.retroreflectivity = 0.75;
			expect( object.retroreflectivity ).toBe( 0.75 );

		} );

		test( 'copy copies retroreflectivity', () => {

			const source = new MeshPhysicalMaterial( { retroreflectivity: 0.5 } );
			const object = new MeshPhysicalMaterial();

			object.copy( source );

			expect( object.retroreflectivity ).toBe( 0.5 );

		} );

		test( 'fromJSON restores retroreflectivity', () => {

			const source = new MeshPhysicalMaterial( { retroreflectivity: 0.25 } );
			const json = source.toJSON();
			const object = new MeshPhysicalMaterial();

			object.fromJSON( json );

			expect( json.retroreflectivity ).toBe( 0.25 );
			expect( object.retroreflectivity ).toBe( 0.25 );

		} );

		test( 'isMeshPhysicalMaterial', () => {

			const object = new MeshPhysicalMaterial();
			expect( object.isMeshPhysicalMaterial ).toBeTruthy();

		} );

	} );

} );

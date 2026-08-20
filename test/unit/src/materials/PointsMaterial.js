import { describe, test, expect } from 'vitest';
import { PointsMaterial } from '@src/materials/PointsMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'PointsMaterial', () => {

		test( 'Extending', () => {

			const object = new PointsMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PointsMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new PointsMaterial();
			expect( object.type === 'PointsMaterial' ).toBeTruthy();

		} );

		test( 'isPointsMaterial', () => {

			const object = new PointsMaterial();
			expect( object.isPointsMaterial ).toBeTruthy();

		} );

	} );

} );

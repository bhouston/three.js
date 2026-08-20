import { describe, test, expect } from 'vitest';
import { SpriteMaterial } from '@src/materials/SpriteMaterial.js';
import { Material } from '@src/materials/Material.js';

describe( 'Materials', () => {

	describe( 'SpriteMaterial', () => {

		test( 'Extending', () => {

			const object = new SpriteMaterial();
			expect( object instanceof Material ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new SpriteMaterial();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new SpriteMaterial();
			expect( object.type === 'SpriteMaterial' ).toBeTruthy();

		} );

		test( 'isSpriteMaterial', () => {

			const object = new SpriteMaterial();
			expect( object.isSpriteMaterial ).toBeTruthy();

		} );

	} );

} );

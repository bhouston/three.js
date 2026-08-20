import { describe, test, expect } from 'vitest';
import { Material } from '@src/materials/Material.js';
import { EventDispatcher } from '@src/core/EventDispatcher.js';

describe( 'Materials', () => {

	describe( 'Material', () => {

		test( 'Extending', () => {

			const object = new Material();
			expect( object instanceof EventDispatcher ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Material();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Material();
			expect( object.type === 'Material' ).toBeTruthy();

		} );

		test( 'isMaterial', () => {

			const object = new Material();
			expect( object.isMaterial ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new Material();
			object.dispose();

		} );

	} );

} );

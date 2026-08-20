import { describe, test, expect } from 'vitest';
import { Group } from '@src/objects/Group.js';
import { Object3D } from '@src/core/Object3D.js';

describe( 'Objects', () => {

	describe( 'Group', () => {

		test( 'Extending', () => {

			const group = new Group();
			expect( group instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Group();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Group();
			expect( object.type === 'Group' ).toBeTruthy();

		} );

		test( 'isGroup', () => {

			const object = new Group();
			expect( object.isGroup ).toBeTruthy();

		} );

	} );

} );

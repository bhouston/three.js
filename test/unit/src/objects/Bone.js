import { describe, test, expect } from 'vitest';
import { Bone } from '@src/objects/Bone.js';
import { Object3D } from '@src/core/Object3D.js';

describe( 'Objects', () => {

	describe( 'Bone', () => {

		test( 'Extending', () => {

			const bone = new Bone();
			expect( bone instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Bone();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Bone();
			expect( object.type === 'Bone' ).toBeTruthy();

		} );

		test( 'isBone', () => {

			const object = new Bone();
			expect( object.isBone ).toBeTruthy();

		} );

	} );

} );

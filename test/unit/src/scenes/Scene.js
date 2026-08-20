import { describe, test, expect } from 'vitest';
import { Scene } from '@src/scenes/Scene.js';
import { Object3D } from '@src/core/Object3D.js';

describe( 'Scenes', () => {

	describe( 'Scene', () => {

		test( 'Extending', () => {

			const object = new Scene();
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Scene();
			expect( object ).toBeTruthy();

		} );

		test( 'isScene', () => {

			const object = new Scene();
			expect( object.isScene ).toBeTruthy();

		} );

	} );

} );

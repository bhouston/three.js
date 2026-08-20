import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';
import { Sprite } from '@src/objects/Sprite.js';

describe( 'Objects', () => {

	describe( 'Sprite', () => {

		test( 'Extending', () => {

			const sprite = new Sprite();
			expect( sprite instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Sprite();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Sprite();
			expect( object.type === 'Sprite' ).toBeTruthy();

		} );

		test( 'isSprite', () => {

			const object = new Sprite();
			expect( object.isSprite ).toBeTruthy();

		} );

	} );

} );

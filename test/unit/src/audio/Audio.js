import { describe, test, expect } from 'vitest';
import { Audio } from '@src/audio/Audio.js';
import { Object3D } from '@src/core/Object3D.js';

describe( 'Audios', () => {

	describe( 'Audio', () => {

		function mockListener() {

			return {
				context: {
					createGain: () => {

						return {
							connect: () => {},
						};

					}
				},
				getInput: () => {},
			};

		}

		test( 'Extending', () => {

			const listener = mockListener();
			const object = new Audio( listener );
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const listener = mockListener();
			const object = new Audio( listener );
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const listener = mockListener();
			const object = new Audio( listener );
			expect( object.type === 'Audio' ).toBeTruthy();

		} );

	} );

} );

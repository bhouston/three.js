import { describe, test, expect, beforeAll, afterAll } from 'vitest';
import { AudioListener } from '@src/audio/AudioListener.js';
import { Object3D } from '@src/core/Object3D.js';

describe( 'Audios', () => {

	describe( 'AudioListener', () => {

		// the node lane has no global `window`, so the AudioListener's default
		// AudioContext.getContext() call needs a mock AudioContext on window
		beforeAll( () => {

			global.window = {
				AudioContext: function () {

					return {
						createGain: () => {

							return {
								connect: () => {},
							};

						}
					};

				},
			};

		} );

		afterAll( () => {

			global.window = undefined;

		} );

		test( 'Extending', () => {

			const object = new AudioListener();
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new AudioListener();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new AudioListener();
			expect( object.type === 'AudioListener' ).toBeTruthy();

		} );

	} );

} );

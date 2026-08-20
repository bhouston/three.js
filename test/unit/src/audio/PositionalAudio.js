import { describe, test, expect } from 'vitest';
import { PositionalAudio } from '@src/audio/PositionalAudio.js';
import { Audio } from '@src/audio/Audio.js';

describe( 'Audios', () => {

	describe( 'PositionalAudio', () => {

		function mock3DListener() {

			return {
				context: {
					createGain: () => {

						return {
							connect: () => {},
						};

					},
					createPanner: () => {

						return {
							connect: () => {},
						};

					}

				},
				getInput: () => {},
			};

		}

		test( 'Extending', () => {

			const listener = mock3DListener();
			const object = new PositionalAudio( listener );
			expect( object instanceof Audio ).toBe( true );

		} );

		test( 'Instancing', () => {

			const listener = mock3DListener();
			const object = new PositionalAudio( listener );
			expect( object ).toBeTruthy();

		} );

	} );

} );

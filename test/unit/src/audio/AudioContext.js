import { describe, test, expect, beforeAll, afterAll } from 'vitest';
import { AudioContext } from '@src/audio/AudioContext.js';

describe( 'Audios', () => {

	describe( 'AudioContext', () => {

		// the node lane has no global `window`, so AudioContext.getContext()
		// needs a mock AudioContext on window to construct against
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

		test( 'getContext', () => {

			const context = AudioContext.getContext();
			expect( context instanceof Object ).toBe( true );

		} );

		test( 'setContext', () => {

			AudioContext.setContext( new window.AudioContext() );
			const context = AudioContext.getContext();
			expect( context instanceof Object ).toBe( true );

		} );

	} );

} );

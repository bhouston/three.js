import { describe, test, expect } from 'vitest';
import { BooleanKeyframeTrack } from '@src/animation/tracks/BooleanKeyframeTrack.js';

import { KeyframeTrack } from '@src/animation/KeyframeTrack.js';

describe( 'Animation', () => {

	describe( 'Tracks', () => {

		describe( 'BooleanKeyframeTrack', () => {

			const parameters = {
				name: '.visible',
				times: [ 0, 1 ],
				values: [ true, false ],
			};

			test( 'Extending', () => {

				const object = new BooleanKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object instanceof KeyframeTrack ).toBe( true );

			} );

			test( 'Instancing', () => {

				// name, times, values
				const object = new BooleanKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object ).toBeTruthy();

			} );

		} );

	} );

} );

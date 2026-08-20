import { describe, test, expect } from 'vitest';
import { StringKeyframeTrack } from '@src/animation/tracks/StringKeyframeTrack.js';

import { KeyframeTrack } from '@src/animation/KeyframeTrack.js';

describe( 'Animation', () => {

	describe( 'Tracks', () => {

		describe( 'StringKeyframeTrack', () => {

			const parameters = {
				name: '.name',
				times: [ 0, 1 ],
				values: [ 'foo', 'bar' ],
			};

			test( 'Extending', () => {

				const object = new StringKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object instanceof KeyframeTrack ).toBe( true );

			} );

			test( 'Instancing', () => {

				// name, times, values
				const object = new StringKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object ).toBeTruthy();

			} );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { NumberKeyframeTrack } from '@src/animation/tracks/NumberKeyframeTrack.js';

import { KeyframeTrack } from '@src/animation/KeyframeTrack.js';

describe( 'Animation', () => {

	describe( 'Tracks', () => {

		describe( 'NumberKeyframeTrack', () => {

			const parameters = {
				name: '.material.opacity',
				times: [ 0, 1 ],
				values: [ 0, 0.5 ],
				interpolation: NumberKeyframeTrack.DefaultInterpolation
			};

			test( 'Extending', () => {

				const object = new NumberKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object instanceof KeyframeTrack ).toBe( true );

			} );

			test( 'Instancing', () => {

				// name, times, values
				const object = new NumberKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object ).toBeTruthy();

				// name, times, values, interpolation
				const object_all = new NumberKeyframeTrack( parameters.name, parameters.times, parameters.values, parameters.interpolation );
				expect( object_all ).toBeTruthy();

			} );

		} );

	} );

} );

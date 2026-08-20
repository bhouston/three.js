import { describe, test, expect } from 'vitest';
import { ColorKeyframeTrack } from '@src/animation/tracks/ColorKeyframeTrack.js';

import { KeyframeTrack } from '@src/animation/KeyframeTrack.js';

describe( 'Animation', () => {

	describe( 'Tracks', () => {

		describe( 'ColorKeyframeTrack', () => {

			const parameters = {
				name: '.material.diffuse',
				times: [ 0, 1 ],
				values: [ 0, 0.5, 1.0 ],
				interpolation: ColorKeyframeTrack.DefaultInterpolation
			};

			test( 'Extending', () => {

				const object = new ColorKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object instanceof KeyframeTrack ).toBe( true );

			} );

			test( 'Instancing', () => {

				// name, times, values
				const object = new ColorKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object ).toBeTruthy();

				// name, times, values, interpolation
				const object_all = new ColorKeyframeTrack( parameters.name, parameters.times, parameters.values, parameters.interpolation );
				expect( object_all ).toBeTruthy();

			} );

		} );

	} );

} );

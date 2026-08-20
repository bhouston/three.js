import { describe, test, expect } from 'vitest';
import { VectorKeyframeTrack } from '@src/animation/tracks/VectorKeyframeTrack.js';

import { KeyframeTrack } from '@src/animation/KeyframeTrack.js';

describe( 'Animation', () => {

	describe( 'Tracks', () => {

		describe( 'VectorKeyframeTrack', () => {

			const parameters = {
				name: '.force',
				times: [ 0 ],
				values: [ 0.5, 0.5, 0.5 ],
				interpolation: VectorKeyframeTrack.DefaultInterpolation
			};

			test( 'Extending', () => {

				const object = new VectorKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object instanceof KeyframeTrack ).toBe( true );

			} );

			test( 'Instancing', () => {

				// name, times, values
				const object = new VectorKeyframeTrack( parameters.name, parameters.times, parameters.values );
				expect( object ).toBeTruthy();

				// name, times, values, interpolation
				const object_all = new VectorKeyframeTrack( parameters.name, parameters.times, parameters.values, parameters.interpolation );
				expect( object_all ).toBeTruthy();

			} );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { NumberKeyframeTrack } from '@src/animation/tracks/NumberKeyframeTrack.js';

import { KeyframeTrack } from '@src/animation/KeyframeTrack.js';
import { CONSOLE_LEVEL } from '@test-utils/console-wrapper.js';

describe( 'Animation', () => {

	describe( 'KeyframeTrack', () => {

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

		test( 'validate', () => {

			const validTrack = new NumberKeyframeTrack( '.material.opacity', [ 0, 1 ], [ 0, 0.5 ] );
			const invalidTrack = new NumberKeyframeTrack( '.material.opacity', [ 0, 1 ], [ 0, NaN ] );

			expect( validTrack.validate() ).toBeTruthy();

			console.level = CONSOLE_LEVEL.OFF;
			expect( invalidTrack.validate() ).toBeFalsy();
			console.level = CONSOLE_LEVEL.DEFAULT;

		} );

		test( 'optimize', () => {

			const track = new NumberKeyframeTrack( '.material.opacity', [ 0, 1, 2, 3, 4 ], [ 0, 0, 0, 0, 1 ] );

			expect( track.values.length ).toBe( 5 );

			track.optimize();

			expect( Array.from( track.times ) ).toSmartEqual( [ 0, 3, 4 ] );
			expect( Array.from( track.values ) ).toSmartEqual( [ 0, 0, 1 ] );

		} );

	} );

} );

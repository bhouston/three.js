import { describe, test, expect } from 'vitest';

import { Clock } from '@src/core/Clock.js';

describe( 'Core', () => {

	describe( 'Clock', () => {

		function mockPerformance() {

			const reference = ( typeof global !== 'undefined' ) ? global : self;

			reference.performance = {
				deltaTime: 0,

				next: function ( delta ) {

					this.deltaTime += delta;

				},

				now: function () {

					return this.deltaTime;

				}

			};

		}

		test( 'Instancing', () => {

			// no params
			const object = new Clock();
			expect( object ).toBeTruthy();

			// autostart
			const object_all = new Clock( false );
			expect( object_all ).toBeTruthy();

		} );

		test( 'clock with performance', () => {

			if ( typeof performance === 'undefined' ) {

				return;

			}

			mockPerformance();

			const clock = new Clock( false );

			clock.start();

			performance.next( 123 );
			expect( clock.getElapsedTime() ).toNumEqual( 0.123 );

			performance.next( 100 );
			expect( clock.getElapsedTime() ).toNumEqual( 0.223 );

			clock.stop();

			performance.next( 1000 );
			expect( clock.getElapsedTime() ).toNumEqual( 0.223 );

		} );

	} );

} );

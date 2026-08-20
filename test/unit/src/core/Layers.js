import { describe, test, expect } from 'vitest';

import { Layers } from '@src/core/Layers.js';

describe( 'Core', () => {

	describe( 'Layers', () => {

		test( 'Instancing', () => {

			const object = new Layers();
			expect( object ).toBeTruthy();

		} );

		test( 'set', () => {

			const a = new Layers();

			for ( let i = 0; i < 31; i ++ ) {

				a.set( i );
				expect( a.mask ).toBe( Math.pow( 2, i ) );

			}

		} );

		test( 'enable', () => {

			const a = new Layers();

			a.set( 0 );
			a.enable( 0 );
			expect( a.mask ).toBe( 1 );

			a.set( 0 );
			a.enable( 1 );
			expect( a.mask ).toBe( 3 );

			a.set( 1 );
			a.enable( 0 );
			expect( a.mask ).toBe( 3 );

			a.set( 1 );
			a.enable( 1 );
			expect( a.mask ).toBe( 2 );

		} );

		test( 'toggle', () => {

			const a = new Layers();

			a.set( 0 );
			a.toggle( 0 );
			expect( a.mask ).toBe( 0 );

			a.set( 0 );
			a.toggle( 1 );
			expect( a.mask ).toBe( 3 );

			a.set( 1 );
			a.toggle( 0 );
			expect( a.mask ).toBe( 3 );

			a.set( 1 );
			a.toggle( 1 );
			expect( a.mask ).toBe( 0 );

		} );

		test( 'disable', () => {

			const a = new Layers();

			a.set( 0 );
			a.disable( 0 );
			expect( a.mask ).toBe( 0 );

			a.set( 0 );
			a.disable( 1 );
			expect( a.mask ).toBe( 1 );

			a.set( 1 );
			a.disable( 0 );
			expect( a.mask ).toBe( 2 );

			a.set( 1 );
			a.disable( 1 );
			expect( a.mask ).toBe( 0 );

		} );

		test( 'test', () => {

			const a = new Layers();
			const b = new Layers();

			expect( a.test( b ) ).toBeTruthy();

			a.set( 1 );
			expect( a.test( b ) ).toBeFalsy();

			b.toggle( 1 );
			expect( a.test( b ) ).toBeTruthy();

		} );

		test( 'isEnabled', () => {

			const a = new Layers();

			a.enable( 1 );
			expect( a.isEnabled( 1 ) ).toBeTruthy();

			a.enable( 2 );
			expect( a.isEnabled( 2 ) ).toBeTruthy();

			a.toggle( 1 );
			expect( a.isEnabled( 1 ) ).toBeFalsy();
			expect( a.isEnabled( 2 ) ).toBeTruthy();

		} );

	} );

} );

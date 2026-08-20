import { describe, test, expect } from 'vitest';
import { ArcCurve } from '@src/extras/curves/ArcCurve.js';

import { EllipseCurve } from '@src/extras/curves/EllipseCurve.js';

describe( 'Extras', () => {

	describe( 'Curves', () => {

		describe( 'ArcCurve', () => {

			test( 'Extending', () => {

				const object = new ArcCurve();
				expect( object instanceof EllipseCurve ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new ArcCurve();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new ArcCurve();
				expect( object.type === 'ArcCurve' ).toBeTruthy();

			} );

			test( 'isArcCurve', () => {

				const object = new ArcCurve();
				expect( object.isArcCurve ).toBeTruthy();

			} );

		} );

	} );

} );

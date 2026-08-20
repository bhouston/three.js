import { describe, test, expect } from 'vitest';
import { Color, ColorManagement, LinearSRGBColorSpace } from 'three';
import {
	DisplayP3ColorSpace,
	DisplayP3ColorSpaceImpl,
	LinearDisplayP3ColorSpace,
	LinearDisplayP3ColorSpaceImpl,
	LinearRec2020ColorSpace,
	LinearRec2020ColorSpaceImpl
} from '../../../../examples/jsm/math/ColorSpaces.js';

// Reference: https://apps.colorjs.io/convert/

describe( 'Maths', () => {

	describe( 'ColorSpaces', () => {

		ColorManagement.define( {

			[ DisplayP3ColorSpace ]: DisplayP3ColorSpaceImpl,
			[ LinearDisplayP3ColorSpace ]: LinearDisplayP3ColorSpaceImpl,
			[ LinearRec2020ColorSpace ]: LinearRec2020ColorSpaceImpl

		} );

		test( 'DisplayP3ColorSpace', () => {

			const c = new Color().setRGB( 0.3, 0.5, 0.7 );

			ColorManagement.convert( c, LinearSRGBColorSpace, DisplayP3ColorSpace );

			expect( c.r.toFixed( 3 ) ).toBe( '0.614' );
			expect( c.g.toFixed( 3 ) ).toBe( '0.731' );
			expect( c.b.toFixed( 3 ) ).toBe( '0.843' );

			c.setRGB( 1.0, 0.5, 0.01, DisplayP3ColorSpace );

			expect( c.r.toFixed( 3 ) ).toBe( '1.177' );
			expect( c.g.toFixed( 3 ) ).toBe( '0.181' );
			expect( c.b.toFixed( 3 ) ).toBe( '-0.036' );

			expect( c.getStyle( DisplayP3ColorSpace ) ).toBe( 'color(display-p3 1.000 0.500 0.010)' );

		} );

		test( 'LinearDisplayP3ColorSpace', () => {

			const c = new Color().setRGB( 0.3, 0.5, 0.7 );

			ColorManagement.convert( c, LinearSRGBColorSpace, LinearDisplayP3ColorSpace );

			expect( c.r.toFixed( 3 ) ).toBe( '0.336' );
			expect( c.g.toFixed( 3 ) ).toBe( '0.493' );
			expect( c.b.toFixed( 3 ) ).toBe( '0.679' );

			c.setRGB( 1.0, 0.5, 0.01, LinearDisplayP3ColorSpace );

			expect( c.r.toFixed( 3 ) ).toBe( '1.112' );
			expect( c.g.toFixed( 3 ) ).toBe( '0.479' );
			expect( c.b.toFixed( 3 ) ).toBe( '-0.048' );

			expect( c.getStyle( LinearDisplayP3ColorSpace ) ).toBe( 'color(display-p3-linear 1.000 0.500 0.010)' );

		} );

		test( 'LinearRec2020ColorSpace', () => {

			const c = new Color().setRGB( 0.3, 0.5, 0.7 );

			ColorManagement.convert( c, LinearSRGBColorSpace, LinearRec2020ColorSpace );

			expect( c.r.toFixed( 3 ) ).toBe( '0.383' );
			expect( c.g.toFixed( 3 ) ).toBe( '0.488' );
			expect( c.b.toFixed( 3 ) ).toBe( '0.676' );

			c.setRGB( 1.0, 0.5, 0.01, LinearRec2020ColorSpace );

			expect( c.r.toFixed( 3 ) ).toBe( '1.366' );
			expect( c.g.toFixed( 3 ) ).toBe( '0.442' );
			expect( c.b.toFixed( 3 ) ).toBe( '-0.057' );

			expect( c.getStyle( LinearRec2020ColorSpace ) ).toBe( 'color(rec2020-linear 1.000 0.500 0.010)' );

		} );

	} );

} );

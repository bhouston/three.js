import { describe, test, expect, afterEach } from 'vitest';
import { Color } from '@src/math/Color.js';
import { ColorManagement } from '@src/math/ColorManagement.js';
import { eps } from '@test-utils/math-constants.js';
import { CONSOLE_LEVEL } from '@test-utils/console-wrapper.js';
import { SRGBColorSpace } from '@src/constants.js';

describe( 'Maths', () => {

	describe( 'Color', () => {

		const colorManagementEnabled = ColorManagement.enabled;

		afterEach( () => {

			ColorManagement.enabled = colorManagementEnabled;

		} );

		test( 'Instancing', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			// default ctor
			let c = new Color();
			expect( c.r ).toBeTruthy();
			expect( c.g ).toBeTruthy();
			expect( c.b ).toBeTruthy();

			// rgb ctor
			c = new Color( 1, 1, 1 );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g == 1 ).toBeTruthy();
			expect( c.b == 1 ).toBeTruthy();

		} );

		test( 'Color.NAMES', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			expect( Color.NAMES.aliceblue == 0xF0F8FF ).toBeTruthy();

		} );

		test( 'isColor', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color();
			expect( a.isColor === true ).toBeTruthy();

			const b = new Object();
			expect( ! b.isColor ).toBeTruthy();

		} );

		test( 'set', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color();
			const b = new Color( 0.5, 0, 0 );
			const c = new Color( 0xFF0000 );
			const d = new Color( 0, 1.0, 0 );
			const e = new Color( 0.5, 0.5, 0.5 );

			a.set( b );
			expect( a.equals( b ) ).toBeTruthy();

			a.set( 0xFF0000 );
			expect( a.equals( c ) ).toBeTruthy();

			a.set( 'rgb(0,255,0)' );
			expect( a.equals( d ) ).toBeTruthy();

			a.set( 0.5, 0.5, 0.5 );
			expect( a.equals( e ) ).toBeTruthy();

		} );

		test( 'setScalar', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setScalar( 0.5 );
			expect( c.r == 0.5 ).toBeTruthy();
			expect( c.g == 0.5 ).toBeTruthy();
			expect( c.b == 0.5 ).toBeTruthy();

		} );

		test( 'setHex', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setHex( 0xFA8072 );
			expect( c.getHex() == 0xFA8072 ).toBeTruthy();
			expect( c.r == 0xFA / 0xFF ).toBeTruthy();
			expect( c.g == 0x80 / 0xFF ).toBeTruthy();
			expect( c.b == 0x72 / 0xFF ).toBeTruthy();

		} );

		test( 'setRGB', () => {

			ColorManagement.enabled = true;

			const c = new Color();

			c.setRGB( 0.3, 0.5, 0.7 );

			expect( c.r ).toBe( 0.3 );
			expect( c.g ).toBe( 0.5 );
			expect( c.b ).toBe( 0.7 );

			c.setRGB( 0.3, 0.5, 0.7, SRGBColorSpace );

			expect( c.r.toFixed( 3 ) ).toBe( '0.073' );
			expect( c.g.toFixed( 3 ) ).toBe( '0.214' );
			expect( c.b.toFixed( 3 ) ).toBe( '0.448' );

		} );

		test( 'setHSL', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			const hsl = { h: 0, s: 0, l: 0 };
			c.setHSL( 0.75, 1.0, 0.25 );
			c.getHSL( hsl );

			expect( hsl.h == 0.75 ).toBeTruthy();
			expect( hsl.s == 1.00 ).toBeTruthy();
			expect( hsl.l == 0.25 ).toBeTruthy();

		} );

		test( 'setStyle', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color();

			let b = new Color( 8 / 255, 25 / 255, 178 / 255 );
			a.setStyle( 'rgb(8,25,178)' );
			expect( a.equals( b ) ).toBeTruthy();

			b = new Color( 8 / 255, 25 / 255, 178 / 255 );
			a.setStyle( 'rgba(8,25,178,200)' );
			expect( a.equals( b ) ).toBeTruthy();

			let hsl = { h: 0, s: 0, l: 0 };
			a.setStyle( 'hsl(270,50%,75%)' );
			a.getHSL( hsl );
			expect( hsl.h == 0.75 ).toBeTruthy();
			expect( hsl.s == 0.5 ).toBeTruthy();
			expect( hsl.l == 0.75 ).toBeTruthy();

			hsl = { h: 0, s: 0, l: 0 };
			a.setStyle( 'hsl(270,50%,75%)' );
			a.getHSL( hsl );
			expect( hsl.h == 0.75 ).toBeTruthy();
			expect( hsl.s == 0.5 ).toBeTruthy();
			expect( hsl.l == 0.75 ).toBeTruthy();

			a.setStyle( '#F8A' );
			expect( a.r == 0xFF / 255 ).toBeTruthy();
			expect( a.g == 0x88 / 255 ).toBeTruthy();
			expect( a.b == 0xAA / 255 ).toBeTruthy();

			a.setStyle( '#F8ABC1' );
			expect( a.r == 0xF8 / 255 ).toBeTruthy();
			expect( a.g == 0xAB / 255 ).toBeTruthy();
			expect( a.b == 0xC1 / 255 ).toBeTruthy();

			a.setStyle( 'aliceblue' );
			expect( a.r == 0xF0 / 255 ).toBeTruthy();
			expect( a.g == 0xF8 / 255 ).toBeTruthy();
			expect( a.b == 0xFF / 255 ).toBeTruthy();

		} );

		test( 'setColorName', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			const res = c.setColorName( 'aliceblue' );

			expect( c.getHex() == 0xF0F8FF ).toBeTruthy();
			expect( c == res ).toBeTruthy();

		} );

		test( 'clone', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.
			const c = new Color( 'teal' );
			const c2 = c.clone();
			expect( c2.getHex() == 0x008080 ).toBeTruthy();

		} );

		test( 'copy', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 'teal' );
			const b = new Color();
			b.copy( a );
			expect( b.r == 0x00 / 255 ).toBeTruthy();
			expect( b.g == 0x80 / 255 ).toBeTruthy();
			expect( b.b == 0x80 / 255 ).toBeTruthy();

		} );

		test( 'copySRGBToLinear', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			const c2 = new Color();
			c2.setRGB( 0.3, 0.5, 0.9 );
			c.copySRGBToLinear( c2 );
			expect( c.r ).toNumEqual( 0.09 );
			expect( c.g ).toNumEqual( 0.25 );
			expect( c.b ).toNumEqual( 0.81 );

		} );

		test( 'copyLinearToSRGB', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			const c2 = new Color();
			c2.setRGB( 0.09, 0.25, 0.81 );
			c.copyLinearToSRGB( c2 );
			expect( c.r ).toNumEqual( 0.3 );
			expect( c.g ).toNumEqual( 0.5 );
			expect( c.b ).toNumEqual( 0.9 );

		} );

		test( 'convertSRGBToLinear', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setRGB( 0.3, 0.5, 0.9 );
			c.convertSRGBToLinear();
			expect( c.r ).toNumEqual( 0.09 );
			expect( c.g ).toNumEqual( 0.25 );
			expect( c.b ).toNumEqual( 0.81 );

		} );

		test( 'convertLinearToSRGB', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setRGB( 4, 9, 16 );
			c.convertLinearToSRGB();
			expect( c.r ).toNumEqual( 1.82 );
			expect( c.g ).toNumEqual( 2.58 );
			expect( c.b ).toNumEqual( 3.29 );

		} );

		test( 'getHex', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color( 'red' );
			const res = c.getHex();
			expect( res == 0xFF0000 ).toBeTruthy();

		} );

		test( 'getHexString', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color( 'tomato' );
			const res = c.getHexString();
			expect( res == 'ff6347' ).toBeTruthy();

		} );

		test( 'getHSL', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color( 0x80ffff );
			const hsl = { h: 0, s: 0, l: 0 };
			c.getHSL( hsl );

			expect( hsl.h == 0.5 ).toBeTruthy();
			expect( hsl.s == 1.0 ).toBeTruthy();
			expect( ( Math.round( parseFloat( hsl.l ) * 100 ) / 100 ) == 0.75 ).toBeTruthy();

		} );

		test( 'getRGB', () => {

			ColorManagement.enabled = true;

			const c = new Color( 'plum' );
			const t = { r: 0, g: 0, b: 0 };

			c.getRGB( t );

			expect( t.r.toFixed( 3 ) ).toBe( '0.723' );
			expect( t.g.toFixed( 3 ) ).toBe( '0.352' );
			expect( t.b.toFixed( 3 ) ).toBe( '0.723' );

			c.getRGB( t, SRGBColorSpace );

			expect( t.r.toFixed( 3 ) ).toBe( ( 221 / 255 ).toFixed( 3 ) );
			expect( t.g.toFixed( 3 ) ).toBe( ( 160 / 255 ).toFixed( 3 ) );
			expect( t.b.toFixed( 3 ) ).toBe( ( 221 / 255 ).toFixed( 3 ) );

		} );

		test( 'getStyle', () => {

			ColorManagement.enabled = true;

			const c = new Color( 'plum' );

			expect( c.getStyle() ).toBe( 'rgb(221,160,221)' );

		} );

		test( 'offsetHSL', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 'hsl(120,50%,50%)' );
			const b = new Color( 0.36, 0.84, 0.648 );

			a.offsetHSL( 0.1, 0.1, 0.1 );

			expect( Math.abs( a.r - b.r ) <= eps ).toBeTruthy();
			expect( Math.abs( a.g - b.g ) <= eps ).toBeTruthy();
			expect( Math.abs( a.b - b.b ) <= eps ).toBeTruthy();

		} );

		test( 'add', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 0x0000FF );
			const b = new Color( 0xFF0000 );
			const c = new Color( 0xFF00FF );

			a.add( b );

			expect( a.equals( c ) ).toBeTruthy();

		} );

		test( 'addColors', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 0x0000FF );
			const b = new Color( 0xFF0000 );
			const c = new Color( 0xFF00FF );
			const d = new Color();

			d.addColors( a, b );

			expect( d.equals( c ) ).toBeTruthy();

		} );

		test( 'addScalar', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 0.1, 0.0, 0.0 );
			const b = new Color( 0.6, 0.5, 0.5 );

			a.addScalar( 0.5 );

			expect( a.equals( b ) ).toBeTruthy();

		} );

		test( 'sub', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 0x0000CC );
			const b = new Color( 0xFF0000 );
			const c = new Color( 0x0000AA );

			a.sub( b );
			expect( a.getHex() ).toBe( 0xCC );

			a.sub( c );
			expect( a.getHex() ).toBe( 0x22 );

		} );

		test( 'multiply', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 1, 0, 0.5 );
			const b = new Color( 0.5, 1, 0.5 );
			const c = new Color( 0.5, 0, 0.25 );

			a.multiply( b );
			expect( a.equals( c ) ).toBeTruthy();

		} );

		test( 'multiplyScalar', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 0.25, 0, 0.5 );
			const b = new Color( 0.5, 0, 1 );

			a.multiplyScalar( 2 );
			expect( a.equals( b ) ).toBeTruthy();

		} );

		test( 'lerp', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			const c2 = new Color();
			c.setRGB( 0, 0, 0 );
			c.lerp( c2, 0.2 );
			expect( c.r == 0.2 ).toBeTruthy();
			expect( c.g == 0.2 ).toBeTruthy();
			expect( c.b == 0.2 ).toBeTruthy();

		} );

		test( 'equals', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 0.5, 0.0, 1.0 );
			const b = new Color( 0.5, 1.0, 0.0 );

			expect( a.r ).toBe( b.r );
			expect( a.g ).not.toBe( b.g );
			expect( a.b ).not.toBe( b.b );

			expect( a.equals( b ) ).toBeFalsy();
			expect( b.equals( a ) ).toBeFalsy();

			a.copy( b );
			expect( a.r ).toBe( b.r );
			expect( a.g ).toBe( b.g );
			expect( a.b ).toBe( b.b );

			expect( a.equals( b ) ).toBeTruthy();
			expect( b.equals( a ) ).toBeTruthy();

		} );

		test( 'fromArray', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color();
			const array = [ 0.5, 0.6, 0.7, 0, 1, 0 ];

			a.fromArray( array );
			expect( a.r ).toBe( 0.5 );
			expect( a.g ).toBe( 0.6 );
			expect( a.b ).toBe( 0.7 );

			a.fromArray( array, 3 );
			expect( a.r ).toBe( 0 );
			expect( a.g ).toBe( 1 );
			expect( a.b ).toBe( 0 );

		} );

		test( 'toArray', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const r = 0.5, g = 1.0, b = 0.0;
			const a = new Color( r, g, b );

			let array = a.toArray();
			expect( array[ 0 ] ).toBe( r );
			expect( array[ 1 ] ).toBe( g );
			expect( array[ 2 ] ).toBe( b );

			array = [];
			a.toArray( array );
			expect( array[ 0 ] ).toBe( r );
			expect( array[ 1 ] ).toBe( g );
			expect( array[ 2 ] ).toBe( b );

			array = [];
			a.toArray( array, 1 );
			expect( array[ 0 ] ).toBe( undefined );
			expect( array[ 1 ] ).toBe( r );
			expect( array[ 2 ] ).toBe( g );
			expect( array[ 3 ] ).toBe( b );

		} );

		test( 'toJSON', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const a = new Color( 0.0, 0.0, 0.0 );
			const b = new Color( 0.0, 0.5, 0.0 );
			const c = new Color( 1.0, 0.0, 0.0 );
			const d = new Color( 1.0, 1.0, 1.0 );

			expect( a.toJSON() ).toBe( 0x000000 );
			expect( b.toJSON() ).toBe( 0x008000 );
			expect( c.toJSON() ).toBe( 0xFF0000 );
			expect( d.toJSON() ).toBe( 0xFFFFFF );

		} );

		test( 'copyHex', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			const c2 = new Color( 0xF5FFFA );
			c.copy( c2 );
			expect( c.getHex() == c2.getHex() ).toBeTruthy();

		} );

		test( 'copyColorString', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			const c2 = new Color( 'ivory' );
			c.copy( c2 );
			expect( c.getHex() == c2.getHex() ).toBeTruthy();

		} );

		test( 'setWithNum', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.set( 0xFF0000 );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setWithString', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.set( 'silver' );
			expect( c.getHex() == 0xC0C0C0 ).toBeTruthy();

		} );

		test( 'setStyleRGBRed', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'rgb(255,0,0)' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleRGBARed', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();

			console.level = CONSOLE_LEVEL.ERROR;
			c.setStyle( 'rgba(255,0,0,0.5)' );
			console.level = CONSOLE_LEVEL.DEFAULT;

			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleRGBRedWithSpaces', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'rgb( 255 , 0,   0 )' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleRGBARedWithSpaces', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'rgba( 255,  0,  0  , 1 )' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleRGBPercent', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'rgb(100%,50%,10%)' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g == 0.5 ).toBeTruthy();
			expect( c.b == 0.1 ).toBeTruthy();

		} );

		test( 'setStyleRGBAPercent', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();

			console.level = CONSOLE_LEVEL.ERROR;
			c.setStyle( 'rgba(100%,50%,10%, 0.5)' );
			console.level = CONSOLE_LEVEL.DEFAULT;

			expect( c.r == 1 ).toBeTruthy();
			expect( c.g == 0.5 ).toBeTruthy();
			expect( c.b == 0.1 ).toBeTruthy();

		} );

		test( 'setStyleRGBPercentWithSpaces', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'rgb( 100% ,50%  , 10% )' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g == 0.5 ).toBeTruthy();
			expect( c.b == 0.1 ).toBeTruthy();

		} );

		test( 'setStyleRGBAPercentWithSpaces', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();

			console.level = CONSOLE_LEVEL.ERROR;
			c.setStyle( 'rgba( 100% ,50%  ,  10%, 0.5 )' );
			console.level = CONSOLE_LEVEL.DEFAULT;

			expect( c.r == 1 ).toBeTruthy();
			expect( c.g == 0.5 ).toBeTruthy();
			expect( c.b == 0.1 ).toBeTruthy();

		} );

		test( 'setStyleHSLRed', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'hsl(360,100%,50%)' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleHSLARed', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();

			console.level = CONSOLE_LEVEL.ERROR;
			c.setStyle( 'hsla(360,100%,50%,0.5)' );
			console.level = CONSOLE_LEVEL.DEFAULT;

			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleHSLRedWithSpaces', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'hsl(360,  100% , 50% )' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleHSLARedWithSpaces', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();

			console.level = CONSOLE_LEVEL.ERROR;
			c.setStyle( 'hsla( 360,  100% , 50%,  0.5 )' );
			console.level = CONSOLE_LEVEL.DEFAULT;

			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleHSLRedWithDecimals', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'hsl(360,100.0%,50.0%)' );
			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleHSLARedWithDecimals', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();

			console.level = CONSOLE_LEVEL.ERROR;
			c.setStyle( 'hsla(360,100.0%,50.0%,0.5)' );
			console.level = CONSOLE_LEVEL.DEFAULT;

			expect( c.r == 1 ).toBeTruthy();
			expect( c.g === 0 ).toBeTruthy();
			expect( c.b === 0 ).toBeTruthy();

		} );

		test( 'setStyleHexSkyBlue', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( '#87CEEB' );
			expect( c.getHex() == 0x87CEEB ).toBeTruthy();

		} );

		test( 'setStyleHexSkyBlueMixed', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( '#87cEeB' );
			expect( c.getHex() == 0x87CEEB ).toBeTruthy();

		} );

		test( 'setStyleHex2Olive', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( '#F00' );
			expect( c.getHex() == 0xFF0000 ).toBeTruthy();

		} );

		test( 'setStyleHex2OliveMixed', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( '#f00' );
			expect( c.getHex() == 0xFF0000 ).toBeTruthy();

		} );

		test( 'setStyleColorName', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color();
			c.setStyle( 'powderblue' );
			expect( c.getHex() == 0xB0E0E6 ).toBeTruthy();

		} );

		test( 'iterable', () => {

			ColorManagement.enabled = false; // TODO: Update and enable.

			const c = new Color( 0.5, 0.75, 1 );
			const array = [ ...c ];
			expect( array[ 0 ] ).toBe( 0.5 );
			expect( array[ 1 ] ).toBe( 0.75 );
			expect( array[ 2 ] ).toBe( 1 );

		} );

	} );

} );

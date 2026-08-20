import { describe, test, expect } from 'vitest';
import { arrayMin, arrayMax, getTypedArray } from '@src/utils.js';

describe( 'utils', () => {

	test( 'arrayMin', () => {

		expect( arrayMin( [] ) ).toBe( Infinity );
		expect( arrayMin( [ 5 ] ) ).toBe( 5 );
		expect( arrayMin( [ 1, 5, 10 ] ) ).toBe( 1 );
		expect( arrayMin( [ 5, 1, 10 ] ) ).toBe( 1 );
		expect( arrayMin( [ 10, 5, 1 ] ) ).toBe( 1 );
		expect( arrayMax( [ - 0, 0 ] ) ).toBe( - 0 );
		expect( arrayMin( [ - Infinity, 0, Infinity ] ) ).toBe( - Infinity );

	} );

	test( 'arrayMax', () => {

		expect( arrayMax( [] ) ).toBe( - Infinity );
		expect( arrayMax( [ 5 ] ) ).toBe( 5 );
		expect( arrayMax( [ 10, 5, 1 ] ) ).toBe( 10 );
		expect( arrayMax( [ 1, 10, 5 ] ) ).toBe( 10 );
		expect( arrayMax( [ 1, 5, 10 ] ) ).toBe( 10 );
		expect( arrayMax( [ - 0, 0 ] ) ).toBe( - 0 );
		expect( arrayMax( [ - Infinity, 0, Infinity ] ) ).toBe( Infinity );

	} );

	test( 'getTypedArray', () => {

		expect( getTypedArray( 'Int8Array', new ArrayBuffer() ) instanceof Int8Array ).toBeTruthy();
		expect( getTypedArray( 'Uint8Array', new ArrayBuffer() ) instanceof Uint8Array ).toBeTruthy();
		expect( getTypedArray( 'Uint8ClampedArray', new ArrayBuffer() ) instanceof Uint8ClampedArray ).toBeTruthy();
		expect( getTypedArray( 'Int16Array', new ArrayBuffer() ) instanceof Int16Array ).toBeTruthy();
		expect( getTypedArray( 'Uint16Array', new ArrayBuffer() ) instanceof Uint16Array ).toBeTruthy();
		expect( getTypedArray( 'Int32Array', new ArrayBuffer() ) instanceof Int32Array ).toBeTruthy();
		expect( getTypedArray( 'Uint32Array', new ArrayBuffer() ) instanceof Uint32Array ).toBeTruthy();
		expect( getTypedArray( 'Float32Array', new ArrayBuffer() ) instanceof Float32Array ).toBeTruthy();
		expect( getTypedArray( 'Float64Array', new ArrayBuffer() ) instanceof Float64Array ).toBeTruthy();

	} );

} );

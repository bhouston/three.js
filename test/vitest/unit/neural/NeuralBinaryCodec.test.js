import { describe, expect, it } from 'vitest';
import {
	base64FromBytes,
	bytesFromBase64,
	float32ToFloat16,
	float16ToFloat32,
	encodeFloat16Base64,
	decodeFloat16Base64,
	encodeUint8Base64,
	decodeUint8Base64
} from '../../../../examples/jsm/neural/NeuralBinaryCodec.js';

describe( 'Addons > Neural > NeuralBinaryCodec', () => {

	describe( 'base64FromBytes / bytesFromBase64', () => {

		it( 'round-trips small byte arrays', () => {

			const bytes = new Uint8Array( [ 0, 1, 2, 254, 255, 128 ] );
			const decoded = bytesFromBase64( base64FromBytes( bytes ) );
			expect( Array.from( decoded ) ).toEqual( Array.from( bytes ) );

		} );

		it( 'round-trips an empty array', () => {

			const bytes = new Uint8Array( 0 );
			const decoded = bytesFromBase64( base64FromBytes( bytes ) );
			expect( decoded.length ).toBe( 0 );

		} );

		it( 'round-trips a large byte array spanning multiple chunks', () => {

			const bytes = new Uint8Array( 50000 );
			for ( let i = 0; i < bytes.length; i ++ ) bytes[ i ] = i % 256;
			const decoded = bytesFromBase64( base64FromBytes( bytes ) );
			expect( Array.from( decoded ) ).toEqual( Array.from( bytes ) );

		} );

	} );

	describe( 'float32ToFloat16 / float16ToFloat32', () => {

		it( 'round-trips exactly representable values', () => {

			for ( const value of [ 0, 1, - 1, 0.5, - 0.5, 2, 100, - 100 ] ) {

				expect( float16ToFloat32( float32ToFloat16( value ) ) ).toBeCloseTo( value, 5 );

			}

		} );

		it( 'round-trips small fractional values within half-float precision', () => {

			const value = 0.12345;
			const roundTripped = float16ToFloat32( float32ToFloat16( value ) );
			expect( Math.abs( roundTripped - value ) ).toBeLessThan( 1e-3 );

		} );

	} );

	describe( 'encodeFloat16Base64 / decodeFloat16Base64', () => {

		it( 'round-trips a Float32Array within half-float precision', () => {

			const data = new Float32Array( [ 0, 1, - 1, 0.25, 3.5, - 7.125, 0.001 ] );
			const decoded = decodeFloat16Base64( encodeFloat16Base64( data ), data.length );

			expect( decoded.length ).toBe( data.length );

			for ( let i = 0; i < data.length; i ++ ) {

				expect( Math.abs( decoded[ i ] - data[ i ] ) ).toBeLessThan( Math.max( 1e-3, Math.abs( data[ i ] ) * 1e-2 ) );

			}

		} );

		it( 'round-trips an empty array', () => {

			const decoded = decodeFloat16Base64( encodeFloat16Base64( new Float32Array( 0 ) ), 0 );
			expect( decoded.length ).toBe( 0 );

		} );

	} );

	describe( 'encodeUint8Base64 / decodeUint8Base64', () => {

		it( 'round-trips within the 1/255 quantization step', () => {

			const min = - 2;
			const max = 3;
			const data = new Float32Array( [ - 2, - 1, 0, 0.5, 1, 2.9, 3 ] );
			const decoded = decodeUint8Base64( encodeUint8Base64( data, min, max ), min, max, data.length );

			const step = ( max - min ) / 255;

			for ( let i = 0; i < data.length; i ++ ) {

				expect( Math.abs( decoded[ i ] - data[ i ] ) ).toBeLessThanOrEqual( step / 2 + 1e-6 );

			}

		} );

		it( 'clamps out-of-range values to the min/max bounds', () => {

			const min = 0;
			const max = 1;
			const data = new Float32Array( [ - 5, 10 ] );
			const decoded = decodeUint8Base64( encodeUint8Base64( data, min, max ), min, max, data.length );

			expect( decoded[ 0 ] ).toBeCloseTo( min, 5 );
			expect( decoded[ 1 ] ).toBeCloseTo( max, 5 );

		} );

		it( 'handles min === max by mapping every value to zero without dividing by zero', () => {

			const data = new Float32Array( [ 5, 5, 5 ] );
			const decoded = decodeUint8Base64( encodeUint8Base64( data, 5, 5 ), 5, 5, data.length );

			expect( Array.from( decoded ) ).toEqual( [ 5, 5, 5 ] );

		} );

		it( 'round-trips an empty array', () => {

			const decoded = decodeUint8Base64( encodeUint8Base64( new Float32Array( 0 ), 0, 1 ), 0, 1, 0 );
			expect( decoded.length ).toBe( 0 );

		} );

	} );

} );

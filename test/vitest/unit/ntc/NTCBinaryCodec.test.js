import { describe, expect, it } from 'vitest';
import {
	base64FromBytes,
	bytesFromBase64,
	float32ToFloat16,
	float16ToFloat32,
	encodeFloat16Base64,
	decodeFloat16Base64,
	computeZeroPoint,
	encodeUint8Base64,
	decodeUint8Base64
} from '../../../../examples/jsm/ntc/NTCBinaryCodec.js';

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

		it( 'defaulting bits to 8 matches passing bits: 8 explicitly (backward compatibility)', () => {

			const min = - 2;
			const max = 3;
			const data = new Float32Array( [ - 2, - 1, 0, 0.5, 1, 2.9, 3 ] );

			const implicit = decodeUint8Base64( encodeUint8Base64( data, min, max ), min, max, data.length );
			const explicit = decodeUint8Base64( encodeUint8Base64( data, min, max, 8 ), min, max, data.length, 8 );

			expect( Array.from( implicit ) ).toEqual( Array.from( explicit ) );

		} );

		it( 'a lower bit depth uses fewer, coarser levels (larger max round-trip error than 8-bit)', () => {

			const min = - 1;
			const max = 1;
			const data = new Float32Array( 200 );
			for ( let i = 0; i < data.length; i ++ ) data[ i ] = min + ( max - min ) * ( i / ( data.length - 1 ) );

			const errorAtBits = ( bits ) => {

				const decoded = decodeUint8Base64( encodeUint8Base64( data, min, max, bits ), min, max, data.length, bits );
				let maxError = 0;
				for ( let i = 0; i < data.length; i ++ ) maxError = Math.max( maxError, Math.abs( decoded[ i ] - data[ i ] ) );
				return maxError;

			};

			const error8 = errorAtBits( 8 );
			const error4 = errorAtBits( 4 );
			const error2 = errorAtBits( 2 );

			expect( error4 ).toBeGreaterThan( error8 );
			expect( error2 ).toBeGreaterThan( error4 );

			// Roughly matches the expected 1/(2**bits - 1) step-size scaling
			// (within a factor of 2, to allow for where the sampled `data`
			// values happen to fall relative to level boundaries).
			const step8 = ( max - min ) / 255;
			const step4 = ( max - min ) / 15;
			expect( error8 ).toBeLessThanOrEqual( step8 / 2 + 1e-6 );
			expect( error4 ).toBeLessThanOrEqual( step4 / 2 + 1e-6 );

		} );

		it( 'only ever produces bytes within [0, 2**bits - 1]', () => {

			const min = - 5;
			const max = 5;
			const data = new Float32Array( [ - 5, - 3.2, 0, 1.1, 5 ] );

			for ( const bits of [ 2, 4, 6, 8 ] ) {

				const bytes = bytesFromBase64( encodeUint8Base64( data, min, max, bits ) );
				const maxLevel = ( 2 ** bits ) - 1;

				for ( const byte of bytes ) {

					expect( byte ).toBeGreaterThanOrEqual( 0 );
					expect( byte ).toBeLessThanOrEqual( maxLevel );

				}

			}

		} );

	} );

	describe( 'computeZeroPoint', () => {

		it( 'returns 0 for a degenerate (min === max) range', () => {

			expect( computeZeroPoint( 5, 5, 256 ) ).toBe( 0 );

		} );

		it( 'clamps to the valid level range', () => {

			// min = 0 -> zero itself is at the very bottom of the range -> zero point 0.
			expect( computeZeroPoint( 0, 10, 256 ) ).toBe( 0 );
			// max = 0 -> zero is at the very top of the range -> zero point at the last level.
			expect( computeZeroPoint( - 10, 0, 256 ) ).toBe( 255 );

		} );

		it( 'places the zero point proportionally within an asymmetric range', () => {

			// [-2, 3], 256 levels: scale = 5/255, zeroPoint = round(2 / (5/255)) = round(102) = 102.
			expect( computeZeroPoint( - 2, 3, 256 ) ).toBe( 102 );

		} );

	} );

	describe( 'encodeUint8Base64 / decodeUint8Base64 with zeroPreserving', () => {

		it( 'recovers exactly 0 with no rounding error, unlike plain linear quantization at a low bit depth', () => {

			const min = - 2;
			// Deliberately not a "nice" number relative to min/bits (unlike e.g.
			// max: 3 at 4 bits, where 0 happens to land exactly on a
			// plain-quantization level anyway) - this range genuinely does not
			// let plain linear quantization recover 0 exactly.
			const max = 3.7;
			const bits = 4; // 16 levels - coarse enough that plain linear quantization measurably misses zero
			const data = new Float32Array( [ 0 ] );

			const plain = decodeUint8Base64( encodeUint8Base64( data, min, max, bits, false ), min, max, data.length, bits, false );
			const zeroPreserving = decodeUint8Base64( encodeUint8Base64( data, min, max, bits, true ), min, max, data.length, bits, true );

			expect( zeroPreserving[ 0 ] ).toBe( 0 );
			// Sanity: plain linear quantization at this bit depth/range does
			// NOT recover zero exactly, otherwise this test wouldn't actually
			// be distinguishing the two schemes.
			expect( plain[ 0 ] ).not.toBe( 0 );

		} );

		it( 'still round-trips non-zero values within roughly one quantization step', () => {

			const min = - 2;
			const max = 3;
			const bits = 6;
			const data = new Float32Array( [ - 2, - 1, 0.5, 1, 2.9, 3 ] );

			const decoded = decodeUint8Base64( encodeUint8Base64( data, min, max, bits, true ), min, max, data.length, bits, true );
			const step = ( max - min ) / ( ( 2 ** bits ) - 1 );

			for ( let i = 0; i < data.length; i ++ ) {

				// A full step of slack (not step/2) since zero-point quantization
				// can shift the effective range by up to one level relative to
				// plain linear quantization - see computeZeroPoint's doc comment.
				expect( Math.abs( decoded[ i ] - data[ i ] ) ).toBeLessThanOrEqual( step + 1e-6 );

			}

		} );

		it( 'handles min === max by mapping every value back to the constant, without dividing by zero', () => {

			const data = new Float32Array( [ 5, 5, 5 ] );
			const decoded = decodeUint8Base64( encodeUint8Base64( data, 5, 5, 4, true ), 5, 5, data.length, 4, true );

			expect( Array.from( decoded ) ).toEqual( [ 5, 5, 5 ] );

		} );

	} );

} );

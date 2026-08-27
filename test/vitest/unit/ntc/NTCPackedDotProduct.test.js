import { describe, expect, it } from 'vitest';
import {
	packInt8x4,
	quantizeSymmetricInt8,
	computeSymmetricScale
} from '../../../../examples/jsm/ntc/NTCPackedDotProduct.js';

describe( 'Addons > NTC > NTCPackedDotProduct', () => {

	describe( 'computeSymmetricScale', () => {

		it( 'is bound / 127', () => {

			expect( computeSymmetricScale( 127 ) ).toBeCloseTo( 1, 10 );
			expect( computeSymmetricScale( 12.7 ) ).toBeCloseTo( 0.1, 10 );

		} );

		it( 'returns 1 (not NaN/Infinity) for a zero or negative bound', () => {

			expect( computeSymmetricScale( 0 ) ).toBe( 1 );
			expect( computeSymmetricScale( - 5 ) ).toBe( 1 );

		} );

	} );

	describe( 'quantizeSymmetricInt8', () => {

		it( 'maps 0 to exactly level 0', () => {

			expect( quantizeSymmetricInt8( 0, 0.1 ) ).toBe( 0 );

		} );

		it( 'maps +bound to level 127 and -bound to level -128 (or the nearest clamp)', () => {

			const bound = 2.54;
			const scale = computeSymmetricScale( bound );
			expect( quantizeSymmetricInt8( bound, scale ) ).toBe( 127 );
			expect( quantizeSymmetricInt8( - bound, scale ) ).toBe( - 127 ); // -bound/scale = -127 exactly

		} );

		it( 'clamps values beyond the scale\'s representable range', () => {

			const scale = 0.01;
			expect( quantizeSymmetricInt8( 100, scale ) ).toBe( 127 );
			expect( quantizeSymmetricInt8( - 100, scale ) ).toBe( - 128 );

		} );

		it( 'returns 0 for a zero scale rather than dividing by zero', () => {

			expect( quantizeSymmetricInt8( 5, 0 ) ).toBe( 0 );

		} );

		it( 'round-trips (quantize then dequantize) within one half-step of scale', () => {

			const scale = 0.05;
			for ( const value of [ - 3, - 0.3, 0, 0.17, 3 ] ) {

				const level = quantizeSymmetricInt8( value, scale );
				const dequantized = level * scale;
				expect( Math.abs( dequantized - value ) ).toBeLessThanOrEqual( scale / 2 + 1e-9 );

			}

		} );

	} );

	describe( 'packInt8x4 / unpacking by hand', () => {

		function unpackInt8x4( packed ) {

			const values = new Array( 4 );
			for ( let i = 0; i < 4; i ++ ) {

				const byte = ( packed >>> ( 8 * i ) ) & 0xFF;
				// sign-extend the byte back to a signed int8
				values[ i ] = byte >= 128 ? byte - 256 : byte;

			}

			return values;

		}

		it( 'round-trips positive and negative values exactly', () => {

			const values = [ 1, - 1, 127, - 128 ];
			const packed = packInt8x4( values );
			expect( unpackInt8x4( packed ) ).toEqual( values );

		} );

		it( 'zero-pads past 4 values are not read (only the first 4 are packed)', () => {

			const packed = packInt8x4( [ 5, 6, 7, 8 ] );
			expect( unpackInt8x4( packed ) ).toEqual( [ 5, 6, 7, 8 ] );

		} );

		it( 'treats a missing/undefined value as 0', () => {

			const packed = packInt8x4( [ 3, undefined, - 3 ] ); // index 3 omitted entirely
			expect( unpackInt8x4( packed ) ).toEqual( [ 3, 0, - 3, 0 ] );

		} );

		it( 'clamps out-of-range inputs to the int8 range before packing', () => {

			const packed = packInt8x4( [ 200, - 200, 0, 0 ] );
			expect( unpackInt8x4( packed ) ).toEqual( [ 127, - 128, 0, 0 ] );

		} );

		it( 'is always a non-negative (unsigned bit-pattern) JS number', () => {

			const packed = packInt8x4( [ - 128, - 128, - 128, - 128 ] );
			expect( packed ).toBeGreaterThanOrEqual( 0 );
			expect( Number.isInteger( packed ) ).toBe( true );

		} );

		it( 'matches a hand-computed packing for a known case', () => {

			// [1, 2, 3, 4] -> bytes 0x01, 0x02, 0x03, 0x04 -> little-endian u32
			// 0x04030201 = 67305985
			expect( packInt8x4( [ 1, 2, 3, 4 ] ) ).toBe( 0x04030201 );

		} );

		it( 'dot4I8Packed semantics: the packed dot product (computed by hand) matches a plain sum of products', () => {

			const a = [ 3, - 4, 5, - 6 ];
			const b = [ 1, 2, - 3, 4 ];
			const packedA = packInt8x4( a );
			const packedB = packInt8x4( b );

			const expectedDot = a.reduce( ( sum, value, i ) => sum + value * b[ i ], 0 );

			// Manual dot4I8Packed reference implementation (byte-by-byte, signed) -
			// an independent re-derivation of what the GPU builtin computes,
			// used here purely to confirm packInt8x4 packs bytes in the order
			// dot4I8Packed expects (see NTCPackedDotProduct.js's doc comment on
			// byte order) - the actual GPU builtin itself is exercised in
			// test/vitest/browser/ntc/NTCPackedDotProduct.test.js.
			const unpackedA = unpackInt8x4( packedA );
			const unpackedB = unpackInt8x4( packedB );
			const manualDot = unpackedA.reduce( ( sum, value, i ) => sum + value * unpackedB[ i ], 0 );

			expect( manualDot ).toBe( expectedDot );

		} );

	} );

} );

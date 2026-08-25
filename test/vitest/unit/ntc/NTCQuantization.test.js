import { describe, expect, it } from 'vitest';
import {
	QUANTIZATION_SCHEMES,
	DEFAULT_QUANTIZATION_OPTIONS,
	resolveQuantizationConfig
} from '../../../../examples/jsm/ntc/NTCQuantization.js';

describe( 'Addons > Neural > NeuralQuantization', () => {

	describe( 'QUANTIZATION_SCHEMES.none', () => {

		it( 'is an identity forward pass', () => {

			for ( const value of [ - 5, 0, 0.123, 1e6 ] ) {

				expect( QUANTIZATION_SCHEMES.none.quantizeForwardCPU( value, - 1, 1 ) ).toBe( value );

			}

		} );

	} );

	describe( 'QUANTIZATION_SCHEMES.uint8', () => {

		it( 'quantizes to one of 256 discrete levels within [min, max]', () => {

			const lo = - 2;
			const hi = 3;
			const step = ( hi - lo ) / 255;

			for ( const value of [ - 2, - 1, 0, 0.37, 1, 2.9, 3 ] ) {

				const quantized = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( value, lo, hi );
				expect( quantized ).toBeGreaterThanOrEqual( lo - 1e-9 );
				expect( quantized ).toBeLessThanOrEqual( hi + 1e-9 );
				expect( Math.abs( quantized - value ) ).toBeLessThanOrEqual( step / 2 + 1e-9 );

			}

		} );

		it( 'is idempotent (quantizing an already-quantized value is a no-op)', () => {

			const lo = 0;
			const hi = 1;
			const once = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( 0.6789, lo, hi );
			const twice = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( once, lo, hi );
			expect( twice ).toBeCloseTo( once, 9 );

		} );

		it( 'clamps values outside [min, max]', () => {

			expect( QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( - 10, 0, 1 ) ).toBeCloseTo( 0, 9 );
			expect( QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( 10, 0, 1 ) ).toBeCloseTo( 1, 9 );

		} );

		it( 'handles min === max without producing NaN', () => {

			expect( QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU( 5, 5, 5 ) ).toBe( 5 );

		} );

	} );

	describe( 'resolveQuantizationConfig', () => {

		it( 'defaults to the documented defaults when no option is given', () => {

			expect( resolveQuantizationConfig( {} ) ).toEqual( DEFAULT_QUANTIZATION_OPTIONS );
			expect( resolveQuantizationConfig() ).toEqual( DEFAULT_QUANTIZATION_OPTIONS );

		} );

		it( 'accepts an explicit uint8 mode with a fixed range', () => {

			const resolved = resolveQuantizationConfig( { quantization: { mode: 'uint8', range: [ - 1, 1 ] } } );
			expect( resolved.mode ).toBe( 'uint8' );
			expect( resolved.range ).toEqual( [ - 1, 1 ] );
			expect( resolved.target ).toBe( 'latents' );
			expect( resolved.perLevel ).toBe( true );

		} );

		it( 'throws a clear error for an invalid mode', () => {

			expect( () => resolveQuantizationConfig( { quantization: { mode: 'int4' } } ) )
				.toThrow( /quantization\.mode/ );

		} );

		it( 'throws a clear error for an invalid target', () => {

			expect( () => resolveQuantizationConfig( { quantization: { target: 'bogus' } } ) )
				.toThrow( /quantization\.target/ );

		} );

		it( 'throws "not yet implemented" only when a non-latents target is actually used with quantization enabled', () => {

			expect( () => resolveQuantizationConfig( { quantization: { mode: 'uint8', target: 'weights' } } ) )
				.toThrow( /not yet implemented/ );

			// target: 'weights' with mode: 'none' should validate fine since
			// quantization isn't actually enabled.
			expect( () => resolveQuantizationConfig( { quantization: { mode: 'none', target: 'weights' } } ) )
				.not.toThrow();

		} );

		it( 'throws a clear error for an invalid range', () => {

			expect( () => resolveQuantizationConfig( { quantization: { range: [ 1, 0 ] } } ) )
				.toThrow( /quantization\.range/ );
			expect( () => resolveQuantizationConfig( { quantization: { range: 'not-auto' } } ) )
				.toThrow( /quantization\.range/ );

		} );

		it( 'throws a clear error for an invalid perLevel', () => {

			expect( () => resolveQuantizationConfig( { quantization: { perLevel: 'yes' } } ) )
				.toThrow( /quantization\.perLevel/ );

		} );

	} );

} );

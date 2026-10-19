import {
	vec3,
	sRGBTransferEOTF, sRGBTransferOETF
} from 'three/tsl';
import { SRGBToLinear, LinearToSRGB } from '../../../../../src/math/ColorManagement.js';
import { mx_srgb_texture_to_lin_rec709 } from '../../../../../src/nodes/materialx/MaterialXColorTransform.js';
import { gpuTest } from './gpu-test-utils.js';

// sRGB <-> linear-sRGB transfer function coverage. Every expected value below
// comes from SRGBToLinear()/LinearToSRGB() (src/math/ColorManagement.js) --
// the same IEC 61966-2-1 piecewise formula that sRGBTransferEOTF()/
// sRGBTransferOETF() implement in TSL, but an independent, pre-existing
// implementation, not derived by re-running the TSL expressions under test
// (see TSLMath.tests.js's file header for why that matters:
// https://ben3d.ca/blog/the-rise-of-test-theater).
export default QUnit.module( 'Nodes', () => {

	QUnit.module( 'TSL', () => {

		QUnit.module( 'color space functions', () => {

			gpuTest( 'sRGBTransferEOTF() decodes sRGB to linear', ( { assert } ) => {

				// Exact endpoints.
				assert.closeAbs( sRGBTransferEOTF( vec3( 0, 0, 0 ) ), vec3( 0, 0, 0 ), 1e-6, 'EOTF(0) == 0' );
				assert.closeAbs( sRGBTransferEOTF( vec3( 1, 1, 1 ) ), vec3( 1, 1, 1 ), 1e-4, 'EOTF(1) == 1' );

				// Below the linear-segment threshold (0.04045): x / 12.92.
				// sRGBTransferEOTF() is declared (via setLayout) as taking/returning
				// vec3, so scalar inputs must be broadcast to vec3 explicitly --
				// a bare float() input still comes back typed vec3, which a bare
				// float() expected value can't be compared against.
				assert.closeAbs( sRGBTransferEOTF( vec3( 0.02 ) ), vec3( SRGBToLinear( 0.02 ) ), 1e-6, 'EOTF(0.02) uses the linear low-end segment' );

				// Above the threshold: ((x + 0.055) / 1.055) ^ 2.4 -- checked at a
				// well-known reference point (sRGB mid-gray 0.5 -> ~0.2140 linear).
				assert.closeAbs( sRGBTransferEOTF( vec3( 0.5 ) ), vec3( SRGBToLinear( 0.5 ) ), 1e-4, 'EOTF(0.5) uses the power-curve segment' );
				assert.closeAbs( sRGBTransferEOTF( vec3( 0.5 ) ), vec3( 0.21404114 ), 1e-4, 'EOTF(0.5) matches the well-known sRGB mid-gray linear value' );

			} );

			gpuTest( 'sRGBTransferOETF() encodes linear to sRGB', ( { assert } ) => {

				// Exact endpoints.
				assert.closeAbs( sRGBTransferOETF( vec3( 0, 0, 0 ) ), vec3( 0, 0, 0 ), 1e-6, 'OETF(0) == 0' );
				assert.closeAbs( sRGBTransferOETF( vec3( 1, 1, 1 ) ), vec3( 1, 1, 1 ), 1e-4, 'OETF(1) == 1' );

				// Below the linear-segment threshold (0.0031308): x * 12.92.
				assert.closeAbs( sRGBTransferOETF( vec3( 0.001 ) ), vec3( LinearToSRGB( 0.001 ) ), 1e-6, 'OETF(0.001) uses the linear low-end segment' );

				// Above the threshold: 1.055 * x^(1/2.4) - 0.055 -- checked against
				// the same linear mid-gray value used above, in reverse.
				assert.closeAbs( sRGBTransferOETF( vec3( 0.21404114 ) ), vec3( LinearToSRGB( 0.21404114 ) ), 1e-3, 'OETF(0.21404114) round-trips back to sRGB mid-gray 0.5' );
				assert.closeAbs( sRGBTransferOETF( vec3( 0.21404114 ) ), vec3( 0.5 ), 1e-3, 'OETF(0.21404114) matches the well-known sRGB mid-gray value' );

				assert.closeAbs( sRGBTransferOETF( vec3( 1 ) ), vec3( 1 ), 1e-4, 'OETF(1) == 1' );

			} );

			gpuTest( 'extended sRGB preserves signed and out-of-range components', ( { assert } ) => {

				for ( const values of [ [ - 0.02, - 0.04046, - 0.5 ], [ - 2, 0, 2 ], [ - 0.04044, 0.04045, 0.04046 ] ] ) {

					const expected = vec3( ...values.map( SRGBToLinear ) );
					assert.closeAbs( sRGBTransferEOTF( vec3( ...values ) ), expected, 1e-4, `EOTF(${ values })` );
					assert.closeAbs( mx_srgb_texture_to_lin_rec709( vec3( ...values ) ), expected, 1e-4, `MaterialX EOTF(${ values })` );

				}

				for ( const values of [ [ - 0.001, - 0.0031309, - 0.21404114 ], [ - 2, 0, 2 ], [ - 0.0031307, 0.0031308, 0.0031309 ] ] ) {

					assert.closeAbs( sRGBTransferOETF( vec3( ...values ) ), vec3( ...values.map( LinearToSRGB ) ), 1e-4, `OETF(${ values })` );

				}

			} );

			gpuTest( 'sRGBTransferEOTF() and sRGBTransferOETF() are inverses of each other', ( { assert } ) => {

				// Round-trip check using values independently known from the tests
				// above -- not by feeding a value through both functions and
				// comparing to itself with no other reference (that alone would be
				// test theater), but the *shape* of a round-trip is still a useful,
				// additional cross-check once each direction is independently
				// verified against the closed-form formula above.
				const linear = vec3( 0.02, 0.2140411, 0.8 );
				const roundTrip = sRGBTransferEOTF( sRGBTransferOETF( linear ) );

				assert.closeAbs( roundTrip, linear, 1e-3, 'EOTF(OETF(x)) == x' );

			} );

		} );

	} );

} );

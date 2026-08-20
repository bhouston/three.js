import { describe, test, expect } from 'vitest';
import {
	SH_BAND_COMPONENTS,
	SH_BAND_WORDS,
	createGaussianSplatGeometry,
	getSphericalHarmonicsDegree,
	linearToSH0,
	sh0ToLinear,
	sigmoid
} from '../../../../examples/jsm/utils/GaussianSplatUtils.js';
import {
	getSphericalHarmonicsCoefficientLocation,
	packSphericalHarmonicsBand,
	unpackSphericalHarmonicsBand
} from './GaussianSplatTestUtils.js';

const EPS = 1e-6;

function closeTo( actual, expected ) {

	expect( Math.abs( actual - expected ) < EPS ).toBeTruthy();

}

describe( 'Addons', () => {

	describe( 'Utils', () => {

		describe( 'GaussianSplatUtils', () => {

			test( 'maps spherical harmonics coefficients to packed uint words', () => {

				expect( SH_BAND_WORDS ).toEqual( [ 0, 3, 4, 6 ] );

				for ( let degree = 1; degree <= 3; degree ++ ) {

					expect( SH_BAND_WORDS[ degree ] ).toBe( Math.ceil( SH_BAND_COMPONENTS[ degree ] / 4 ) );

				}

				expect( getSphericalHarmonicsCoefficientLocation( 0 ) ).toEqual( { word: 0, shift: 0 } );
				expect( getSphericalHarmonicsCoefficientLocation( 3 ) ).toEqual( { word: 0, shift: 24 } );
				expect( getSphericalHarmonicsCoefficientLocation( 4 ) ).toEqual( { word: 1, shift: 0 } );
				expect( getSphericalHarmonicsCoefficientLocation( 20 ) ).toEqual( { word: 5, shift: 0 } );

			} );

			test( 'converts degree-0 spherical harmonics and linear color', () => {

				closeTo( sh0ToLinear( 0 ), 0.5 );
				closeTo( linearToSH0( 0.5 ), 0 );
				closeTo( sh0ToLinear( linearToSH0( 0.25 ) ), 0.25 );

			} );

			test( 'applies sigmoid activation', () => {

				closeTo( sigmoid( 0 ), 0.5 );
				closeTo( sigmoid( Math.log( 3 ) ), 0.75 );

			} );

			test( 'creates Gaussian splat geometry from packed arrays', () => {

				const data = createGaussianSplatGeometry(
					new Float32Array( [ 1, 2, 3 ] ),
					new Float32Array( [ 4, 0, 0, 9, 0, 16 ] ),
					new Uint8Array( [ 128, 128, 128, 128 ] )
				);

				expect( data.getAttribute( 'position' ).count ).toBe( 1 );
				expect( Array.from( data.getAttribute( 'position' ).array ) ).toEqual( [ 1, 2, 3 ] );
				expect( Array.from( data.getAttribute( 'color' ).array ) ).toEqual( [ 128, 128, 128, 128 ] );
				expect( data.boundingBox !== null ).toBeTruthy();
				expect( data.boundingSphere !== null ).toBeTruthy();

			} );

			test( 'creates Gaussian splat geometry with spherical harmonics attributes', () => {

				const coefficients = new Uint8ClampedArray( [ 129, 130, 131, 132, 133, 134, 135, 136, 137 ] );
				const packedWords = packSphericalHarmonicsBand( coefficients, 1, 1 );
				const data = createGaussianSplatGeometry(
					new Float32Array( [ 1, 2, 3 ] ),
					new Float32Array( [ 4, 0, 0, 9, 0, 16 ] ),
					new Uint8Array( [ 128, 128, 128, 128 ] ),
					{
						sh1: packedWords
					}
				);
				const packed = data.getAttribute( 'sphericalHarmonics1' );

				expect( getSphericalHarmonicsDegree( data ) ).toBe( 1 );
				expect( packed.itemSize ).toBe( SH_BAND_WORDS[ 1 ] );
				expect( packed.array instanceof Uint32Array ).toBeTruthy();
				expect( packed.array ).toBe( packedWords );
				expect( Array.from( unpackSphericalHarmonicsBand( packed.array, 1, 1 ) ) ).toEqual(
					Array.from( coefficients )
				);

			} );

			test( 'requires packed uint32 spherical harmonics on geometry', () => {

				expect( () => {

					createGaussianSplatGeometry(
						new Float32Array( [ 1, 2, 3 ] ),
						new Float32Array( [ 4, 0, 0, 9, 0, 16 ] ),
						new Uint8Array( [ 128, 128, 128, 128 ] ),
						{ sh1: new Float32Array( 9 ) }
					);

				} ).toThrow( /must use packed uint32 words/ );

				expect( () => {

					createGaussianSplatGeometry(
						new Float32Array( [ 1, 2, 3 ] ),
						new Float32Array( [ 4, 0, 0, 9, 0, 16 ] ),
						new Uint8Array( [ 128, 128, 128, 128 ] ),
						{ sh1: new Uint8ClampedArray( 9 ) }
					);

				} ).toThrow( /must use packed uint32 words/ );

			} );

		} );

	} );

} );

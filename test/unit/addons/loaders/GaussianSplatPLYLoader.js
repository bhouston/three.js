import { describe, test, expect } from 'vitest';
import { BufferGeometry } from 'three';

import { GaussianSplatPLYLoader } from '../../../../examples/jsm/loaders/GaussianSplatPLYLoader.js';
import { unpackSphericalHarmonicsBand } from '../utils/GaussianSplatTestUtils.js';

const EPS = 1e-6;

function closeTo( actual, expected ) {

	expect( Math.abs( actual - expected ) < EPS ).toBeTruthy();

}

function createGaussianSplatPLY( shRestCoefficientCount = 0 ) {

	const properties = [
		'property float x',
		'property float y',
		'property float z',
		'property float scale_0',
		'property float scale_1',
		'property float scale_2',
		'property float rot_0',
		'property float rot_1',
		'property float rot_2',
		'property float rot_3',
		'property float f_dc_0',
		'property float f_dc_1',
		'property float f_dc_2',
		'property float opacity'
	];

	for ( let i = 0; i < shRestCoefficientCount; i ++ ) {

		properties.push( `property float f_rest_${ i }` );

	}

	const values = [
		1, 2, 3,
		Math.log( 2 ), Math.log( 3 ), Math.log( 4 ),
		1, 0, 0, 0,
		0, 0, 0,
		0
	];

	for ( let i = 0; i < shRestCoefficientCount; i ++ ) {

		values.push( i / 128 );

	}

	return [
		'ply',
		'format ascii 1.0',
		'element vertex 1',
		...properties,
		'end_header',
		values.join( ' ' )
	].join( '\n' );

}

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'GaussianSplatPLYLoader', () => {

			test( 'parses a Gaussian splat PLY with no spherical harmonics', () => {

				const loader = new GaussianSplatPLYLoader();
				const data = loader.parse( createGaussianSplatPLY() );
				const covariances = data.getAttribute( 'covariance' ).array;

				expect( data instanceof BufferGeometry ).toBeTruthy();
				expect( Array.from( data.getAttribute( 'position' ).array ) ).toEqual( [ 1, 2, 3 ] );
				closeTo( covariances[ 0 ], 4 );
				closeTo( covariances[ 3 ], 9 );
				closeTo( covariances[ 5 ], 16 );
				expect( Array.from( data.getAttribute( 'color' ).array ) ).toEqual( [ 128, 128, 128, 128 ] );

			} );

			test( 'detects spherical harmonics degree from the header and converts f_rest', () => {

				const loader = new GaussianSplatPLYLoader();
				const data = loader.parse( createGaussianSplatPLY( 9 ) );

				expect( Array.from( unpackSphericalHarmonicsBand( data.getAttribute( 'sphericalHarmonics1' ).array, 1, 1 ) ) ).toEqual(
					[ 128, 131, 134, 129, 132, 135, 130, 133, 136 ]
				);

			} );

			test( 'requires no pre-setup for each supported spherical harmonics degree', () => {

				const loader = new GaussianSplatPLYLoader();

				for ( const [ , count ] of [[ 0, 0 ], [ 1, 9 ], [ 2, 24 ], [ 3, 45 ]] ) {

					const data = loader.parse( createGaussianSplatPLY( count ) );
					expect( data.getAttribute( 'position' ).count ).toBe( 1 );

				}

			} );

			test( 'rejects an unsupported number of f_rest coefficients', () => {

				const loader = new GaussianSplatPLYLoader();

				expect( () => loader.parse( createGaussianSplatPLY( 5 ) ) ).toThrow(
					/Unsupported number of f_rest spherical harmonics coefficients/
				);

			} );

			test( 'rejects PLY data missing required Gaussian splat properties', () => {

				const ply = [
					'ply',
					'format ascii 1.0',
					'element vertex 1',
					'property float x',
					'property float y',
					'property float z',
					'end_header',
					'1 2 3'
				].join( '\n' );

				const loader = new GaussianSplatPLYLoader();

				expect( () => loader.parse( ply ) ).toThrow(
					/requires position, scale, rotation, f_dc and opacity properties/
				);

			} );

		} );

	} );

} );

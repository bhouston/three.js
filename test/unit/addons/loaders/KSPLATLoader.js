import { describe, test, expect } from 'vitest';
import { BufferGeometry } from 'three';
import { KSPLATLoader } from '../../../../examples/jsm/loaders/KSPLATLoader.js';
import { unpackSphericalHarmonicsBand } from '../utils/GaussianSplatTestUtils.js';

const EPS = 1e-6;
const HEADER_SIZE_BYTES = 4096;
const SECTION_HEADER_SIZE_BYTES = 1024;
const SH_DEGREE_TO_COMPONENTS = [ 0, 9, 24, 45 ];

function closeTo( actual, expected ) {

	expect( Math.abs( actual - expected ) < EPS ).toBeTruthy();

}

function createKSPLATBuffer( sphericalHarmonicsDegree = 0 ) {

	const degrees = Array.isArray( sphericalHarmonicsDegree ) ? sphericalHarmonicsDegree : [ sphericalHarmonicsDegree ];
	const bytesPerSplat = degrees.map( degree => 44 + SH_DEGREE_TO_COMPONENTS[ degree ] * 4 );
	const sectionHeadersSize = SECTION_HEADER_SIZE_BYTES * degrees.length;
	const buffer = new ArrayBuffer( HEADER_SIZE_BYTES + sectionHeadersSize + bytesPerSplat.reduce( ( sum, size ) => sum + size, 0 ) );
	const view = new DataView( buffer );
	const bytes = new Uint8Array( buffer );

	view.setUint8( 0, 0 );
	view.setUint8( 1, 1 );
	view.setUint32( 4, degrees.length, true );
	view.setUint32( 8, degrees.length, true );
	view.setUint32( 12, degrees.length, true );
	view.setUint32( 16, degrees.length, true );
	view.setUint16( 20, 0, true );

	let dataOffset = HEADER_SIZE_BYTES + sectionHeadersSize;

	for ( let sectionIndex = 0; sectionIndex < degrees.length; sectionIndex ++ ) {

		const degree = degrees[ sectionIndex ];
		const sectionOffset = HEADER_SIZE_BYTES + sectionIndex * SECTION_HEADER_SIZE_BYTES;

		view.setUint32( sectionOffset, 1, true );
		view.setUint32( sectionOffset + 4, 1, true );
		view.setUint32( sectionOffset + 8, 0, true );
		view.setUint32( sectionOffset + 12, 0, true );
		view.setFloat32( sectionOffset + 16, 4, true );
		view.setUint16( sectionOffset + 20, 0, true );
		view.setUint32( sectionOffset + 24, 32767, true );
		view.setUint32( sectionOffset + 32, 0, true );
		view.setUint32( sectionOffset + 36, 0, true );
		view.setUint16( sectionOffset + 40, degree, true );

		view.setFloat32( dataOffset, sectionIndex + 1, true );
		view.setFloat32( dataOffset + 4, 2, true );
		view.setFloat32( dataOffset + 8, 3, true );
		view.setFloat32( dataOffset + 12, 2, true );
		view.setFloat32( dataOffset + 16, 3, true );
		view.setFloat32( dataOffset + 20, 4, true );
		view.setFloat32( dataOffset + 24, 1, true );
		view.setFloat32( dataOffset + 28, 0, true );
		view.setFloat32( dataOffset + 32, 0, true );
		view.setFloat32( dataOffset + 36, 0, true );
		bytes.set( [ 10, 20, 30, 40 ], dataOffset + 40 );

		for ( let i = 0; i < SH_DEGREE_TO_COMPONENTS[ degree ]; i ++ ) {

			view.setFloat32( dataOffset + 44 + i * 4, ( i + 1 ) / 128, true );

		}

		dataOffset += bytesPerSplat[ sectionIndex ];

	}

	return buffer;

}

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'KSPLATLoader', () => {

			test( 'parses uncompressed KSPLAT data', () => {

				const loader = new KSPLATLoader();
				const data = loader.parse( createKSPLATBuffer() );

				const covariances = data.getAttribute( 'covariance' ).array;

				expect( data instanceof BufferGeometry ).toBeTruthy();
				expect( data.getAttribute( 'position' ).count ).toBe( 1 );
				expect( Array.from( data.getAttribute( 'position' ).array ) ).toEqual( [ 1, 2, 3 ] );
				closeTo( covariances[ 0 ], 4 );
				closeTo( covariances[ 1 ], 0 );
				closeTo( covariances[ 2 ], 0 );
				closeTo( covariances[ 3 ], 9 );
				closeTo( covariances[ 4 ], 0 );
				closeTo( covariances[ 5 ], 16 );
				expect( Array.from( data.getAttribute( 'color' ).array ) ).toEqual( [ 10, 20, 30, 40 ] );

			} );

			test( 'parses uncompressed KSPLAT spherical harmonics data', () => {

				const loader = new KSPLATLoader();
				const data = loader.parse( createKSPLATBuffer( 1 ) );

				expect( Array.from( unpackSphericalHarmonicsBand( data.getAttribute( 'sphericalHarmonics1' ).array, 1, 1 ) ) ).toEqual( [ 129, 132, 135, 130, 133, 136, 131, 134, 137 ] );

			} );

			test( 'initializes missing KSPLAT spherical harmonics coefficients to zero', () => {

				const loader = new KSPLATLoader();
				const data = loader.parse( createKSPLATBuffer( [ 0, 1 ] ) );

				expect( Array.from( unpackSphericalHarmonicsBand( data.getAttribute( 'sphericalHarmonics1' ).array, 2, 1 ) ) ).toEqual( [
					128, 128, 128, 128, 128, 128, 128, 128, 128,
					129, 132, 135, 130, 133, 136, 131, 134, 137
				] );

			} );

		} );

	} );

} );

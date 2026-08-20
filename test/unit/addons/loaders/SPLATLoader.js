import { describe, test, expect } from 'vitest';
import { BufferGeometry } from 'three';
import { SPLATLoader } from '../../../../examples/jsm/loaders/SPLATLoader.js';

const EPS = 1e-6;

function closeTo( actual, expected ) {

	expect( Math.abs( actual - expected ) < EPS ).toBeTruthy();

}

function createSplatBuffer() {

	const buffer = new ArrayBuffer( 32 );
	const view = new DataView( buffer );
	const bytes = new Uint8Array( buffer );

	view.setFloat32( 0, 1, true );
	view.setFloat32( 4, 2, true );
	view.setFloat32( 8, 3, true );
	view.setFloat32( 12, 2, true );
	view.setFloat32( 16, 3, true );
	view.setFloat32( 20, 4, true );

	bytes.set( [ 10, 20, 30, 40 ], 24 );
	bytes.set( [ 128, 128, 128, 128 ], 28 );

	return buffer;

}

describe( 'Addons', () => {

	describe( 'Loaders', () => {

		describe( 'SPLATLoader', () => {

			test( 'parses fixed-width .splat data', () => {

				const loader = new SPLATLoader();
				const data = loader.parse( createSplatBuffer() );

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

		} );

	} );

} );

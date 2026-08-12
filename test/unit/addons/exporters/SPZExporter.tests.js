import {
	BufferAttribute,
	BufferGeometry
} from 'three';

import { SPZExporter } from '../../../../examples/jsm/exporters/SPZExporter.js';
import { SPZLoader } from '../../../../examples/jsm/loaders/SPZLoader.js';
import { createGaussianSplatGeometry } from '../../../../examples/jsm/utils/GaussianSplatUtils.js';
import {
	packSphericalHarmonicsBand,
	unpackSphericalHarmonicsBand
} from '../utils/GaussianSplatTestUtils.js';

const EPS = 1e-6;
const SPZ_MAGIC = 0x5053474e;

function closeTo( assert, actual, expected, message ) {

	assert.ok( Math.abs( actual - expected ) < EPS, `${ message }: ${ actual } ~= ${ expected }` );

}

function parseSPZ( buffer ) {

	return new Promise( ( resolve, reject ) => {

		new SPZLoader().parse( buffer, resolve, reject );

	} );

}

function createGeometry( sphericalHarmonicsDegree = 0 ) {

	const sphericalHarmonics = {};

	if ( sphericalHarmonicsDegree >= 1 ) {

		sphericalHarmonics.sh1 = packSphericalHarmonicsBand( new Uint8Array( [
			128, 129, 130,
			131, 132, 133,
			134, 135, 136
		] ), 1, 1 );

	}

	if ( sphericalHarmonicsDegree >= 2 ) {

		sphericalHarmonics.sh2 = packSphericalHarmonicsBand( new Uint8Array( [
			136, 137, 138,
			139, 140, 141,
			142, 143, 144,
			145, 146, 147,
			148, 149, 150
		] ), 1, 2 );

	}

	if ( sphericalHarmonicsDegree >= 3 ) {

		sphericalHarmonics.sh3 = packSphericalHarmonicsBand( new Uint8Array( [
			152, 153, 154,
			155, 156, 157,
			158, 159, 160,
			161, 162, 163,
			164, 165, 166,
			167, 168, 169,
			170, 171, 172
		] ), 1, 3 );

	}

	return createGaussianSplatGeometry(
		new Float32Array( [ 1.5, - 2, 0.25 ] ),
		new Float32Array( [ 1, 0, 0, 1, 0, 1 ] ),
		new Uint8Array( [ 128, 128, 128, 64 ] ),
		sphericalHarmonics
	);

}

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Exporters', () => {

		QUnit.module( 'SPZExporter', () => {

			QUnit.test( 'methods', ( assert ) => {

				const exporter = new SPZExporter();

				assert.ok( exporter instanceof SPZExporter, 'SPZExporter can be instantiated' );
				assert.ok( typeof exporter.parseAsync === 'function', 'parseAsync method exists' );
				assert.ok( typeof exporter.parse === 'function', 'parse method exists' );

			} );

			QUnit.test( 'rejects invalid geometry', async ( assert ) => {

				const exporter = new SPZExporter();
				const geometry = new BufferGeometry();

				geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ 0, 0, 0 ] ), 3 ) );

				await assert.rejects(
					exporter.parseAsync( geometry ),
					/requires position, covariance and color attributes/,
					'missing Gaussian splat attributes are rejected'
				);

			} );

			QUnit.test( 'writes SPZ v4 header and stream table', async ( assert ) => {

				const exporter = new SPZExporter();
				const buffer = await exporter.parseAsync( createGeometry(), { fractionalBits: 4, compressionLevel: 1 } );
				const view = new DataView( buffer );

				assert.strictEqual( view.getUint32( 0, true ), SPZ_MAGIC, 'magic' );
				assert.strictEqual( view.getUint32( 4, true ), 4, 'version' );
				assert.strictEqual( view.getUint32( 8, true ), 1, 'count' );
				assert.strictEqual( view.getUint8( 12 ), 0, 'SH degree' );
				assert.strictEqual( view.getUint8( 13 ), 4, 'fractional bits' );
				assert.strictEqual( view.getUint8( 15 ), 5, 'stream count' );
				assert.strictEqual( view.getUint32( 16, true ), 32, 'TOC offset' );
				assert.ok( view.getBigUint64( 32, true ) > 0n, 'compressed size' );
				assert.strictEqual( view.getBigUint64( 40, true ), 9n, 'positions uncompressed size' );

			} );

			QUnit.test( 'round-trips SPZ v4 degree-0 geometry', async ( assert ) => {

				const exporter = new SPZExporter();
				const result = await parseSPZ( await exporter.parseAsync( createGeometry(), { fractionalBits: 4, compressionLevel: 1 } ) );
				const covariances = result.getAttribute( 'covariance' ).array;

				assert.strictEqual( result.getAttribute( 'position' ).count, 1, 'count' );
				assert.deepEqual( Array.from( result.getAttribute( 'position' ).array ), [ 1.5, - 2, 0.25 ], 'fixed-point centers' );
				closeTo( assert, covariances[ 0 ], 1, 'covariance xx' );
				closeTo( assert, covariances[ 3 ], 1, 'covariance yy' );
				closeTo( assert, covariances[ 5 ], 1, 'covariance zz' );
				assert.deepEqual( Array.from( result.getAttribute( 'color' ).array ), [ 128, 128, 128, 64 ], 'color and alpha' );

			} );

			QUnit.test( 'round-trips SPZ v4 spherical harmonics', async ( assert ) => {

				const exporter = new SPZExporter();
				const result = await parseSPZ( await exporter.parseAsync( createGeometry( 3 ), {
					fractionalBits: 4,
					compressionLevel: 1,
					sh1Bits: 8,
					shRestBits: 8
				} ) );

				assert.deepEqual(
					Array.from( unpackSphericalHarmonicsBand( result.getAttribute( 'sphericalHarmonics1' ).array, 1, 1 ) ),
					[ 128, 129, 130, 131, 132, 133, 134, 135, 136 ],
					'SH1 coefficients'
				);
				assert.deepEqual(
					Array.from( unpackSphericalHarmonicsBand( result.getAttribute( 'sphericalHarmonics2' ).array, 1, 2 ) ),
					[ 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150 ],
					'SH2 coefficients'
				);
				assert.deepEqual(
					Array.from( unpackSphericalHarmonicsBand( result.getAttribute( 'sphericalHarmonics3' ).array, 1, 3 ) ),
					[ 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172 ],
					'SH3 coefficients'
				);

			} );

		} );

	} );

} );

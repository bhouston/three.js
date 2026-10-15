import { UltraHDRLoader } from '../../../../examples/jsm/loaders/UltraHDRLoader.js';
import { FloatType, DataUtils } from '../../../../src/Three.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Loaders', () => {

		QUnit.module( 'UltraHDRLoader', () => {

			const metadata = () => ( {
				version: '1.0', baseRenditionIsHDR: false,
				gainMapMin: [ 0, 0, 0 ], gainMapMax: [ 2, 2, 2 ],
				gamma: [ 1, 1, 1 ], offsetSDR: [ 0, 0, 0 ], offsetHDR: [ 0, 0, 0 ]
			} );

			QUnit.test( 'Full HDR is opt-in and preserves target white', ( assert ) => {

				const loader = new UltraHDRLoader();
				assert.false( loader.fullHDR );
				assert.strictEqual( loader.setFullHDR( true ), loader );
				const result = loader._reconstructFullHDR( metadata(), [ 255, 255, 255, 255 ], 1, 1, [ 255, 255, 255, 255 ], 1, 1 );
				assert.strictEqual( DataUtils.fromHalfFloat( result[ 0 ] ), 4 );
				assert.strictEqual( DataUtils.fromHalfFloat( result[ 3 ] ), 1 );

			} );

			QUnit.test( 'Linear reconstruction uses per-channel gamma and linear offsets', ( assert ) => {

				const loader = new UltraHDRLoader().setFullHDR( true ).setDataType( FloatType );
				const info = metadata();
				info.gamma = [ 1, 2, 0.5 ];
				info.offsetSDR = [ 0.1, 0.2, 0.3 ];
				info.offsetHDR = [ 0.01, 0.02, 0.03 ];
				const result = loader._reconstructFullHDR( info, [ 128, 128, 128, 255 ], 1, 1, [ 64, 128, 192, 255 ], 1, 1 );
				const linear = ( ( 128 / 255 + 0.055 ) / 1.055 ) ** 2.4;
				for ( let c = 0; c < 3; c ++ ) {

					const expected = ( linear + info.offsetSDR[ c ] ) * 2 ** ( 2 * ( ( c + 1 ) * 64 / 255 ) ** ( 1 / info.gamma[ c ] ) ) - info.offsetHDR[ c ];
					assert.true( Math.abs( result[ c ] - expected ) < 1e-6 );

				}

			} );

			QUnit.test( 'Bilinear gain-map resampling operates on encoded samples', ( assert ) => {

				const loader = new UltraHDRLoader().setDataType( FloatType );
				const result = loader._reconstructFullHDR( metadata(), new Uint8Array( 16 ).fill( 255 ), 4, 1, [ 0, 0, 0, 255, 255, 255, 255, 255 ], 2, 1 );
				assert.strictEqual( result[ 0 ], 1 );
				assert.true( Math.abs( result[ 4 ] - Math.sqrt( 2 ) ) < 1e-6 );
				assert.true( Math.abs( result[ 8 ] - Math.sqrt( 8 ) ) < 1e-6 );
				assert.strictEqual( result[ 12 ], 4 );

			} );

			QUnit.test( 'Rejects unsupported metadata and half-float overflow', ( assert ) => {

				const loader = new UltraHDRLoader().setFullHDR( true );
				const info = metadata();
				info.gamma[ 0 ] = 0;
				assert.throws( () => loader._validateFullHDRMetadata( info ), /metadata/ );
				info.gamma[ 0 ] = 1;
				info.baseRenditionIsHDR = true;
				assert.throws( () => loader._validateFullHDRMetadata( info ), /SDR primary/ );
				info.gainMapMax = [ 20, 20, 20 ];
				assert.throws( () => loader._reconstructFullHDR( info, [ 255, 255, 255, 255 ], 1, 1, [ 255, 255, 255, 255 ], 1, 1 ), /representation/ );
				loader.setDataType( FloatType );
				assert.strictEqual( loader._reconstructFullHDR( info, [ 255, 255, 255, 255 ], 1, 1, [ 255, 255, 255, 255 ], 1, 1 )[ 0 ], 1048576 );

			} );

			QUnit.test( 'Reads XMP channel sequences and specification defaults', ( assert ) => {

				const loader = new UltraHDRLoader().setFullHDR( true );
				const info = {};
				const sequence = { getElementsByTagName: () => [ { textContent: '1' }, { textContent: '2' }, { textContent: '3' } ] };
				const node = {
					getAttribute: ( name ) => name === 'hdrgm:Version' ? '1.0' : null,
					getElementsByTagName: ( name ) => name === 'hdrgm:GainMapMax' ? [ sequence ] : []
				};
				loader._parseFullHDRMetadata( node, info );
				loader._validateFullHDRMetadata( info );
				assert.deepEqual( info.gainMapMax, [ 1, 2, 3 ] );
				assert.deepEqual( info.offsetSDR, [ 1 / 64, 1 / 64, 1 / 64 ] );
				assert.deepEqual( info.gamma, [ 1, 1, 1 ] );

			} );

			QUnit.test( 'Image decode errors reach the error callback', async ( assert ) => {

				const loader = new UltraHDRLoader().setFullHDR( true );
				const original = globalThis.createImageBitmap;
				const error = new Error( 'decode failed' );
				globalThis.createImageBitmap = () => Promise.reject( error );
				try {

					await new Promise( ( resolve ) => {

						loader._applyGainmapToSDR( metadata(), new Uint8Array(), new Uint8Array(), () => {

							assert.true( false, 'Must not report successful decoding' );
							resolve();

						}, ( actual ) => {

							assert.strictEqual( actual, error );
							resolve();

						} );

					} );

				} finally {

					globalThis.createImageBitmap = original;

				}

			} );

		} );

	} );

} );

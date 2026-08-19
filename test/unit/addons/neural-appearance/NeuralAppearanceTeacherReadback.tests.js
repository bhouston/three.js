import {
	readSamplePixel,
	renderAndReadTeacher
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherReadback.js';

export default QUnit.module( 'Addons', () => {

	QUnit.module( 'Neural', () => {

		QUnit.module( 'NeuralAppearanceTeacherReadback', () => {

			QUnit.test( 'reads the center teacher pixel of a tile', ( assert ) => {

				const tileSize = 8;
				const atlasColumns = 1;
				const atlasWidth = 8;
				const pixels = new Float32Array( 8 * 8 * 4 );

				for ( let y = 0; y < 8; y ++ ) {

					for ( let x = 0; x < 8; x ++ ) {

						const offset = ( y * 8 + x ) * 4;
						pixels[ offset ] = Math.pow( x / 7, 2 );
						pixels[ offset + 1 ] = Math.pow( y / 7, 2 );

					}

				}

				const point = readSamplePixel( pixels, 0, atlasColumns, atlasWidth, tileSize );

				assert.ok( Number.isFinite( point[ 0 ] ), 'reads a finite red channel' );
				assert.ok( Number.isFinite( point[ 1 ] ), 'reads a finite green channel' );
				assert.strictEqual( point[ 2 ], 0, 'preserves a constant channel' );

			} );

			QUnit.test( 'rejects non-half-float teacher readback', async ( assert ) => {

				const renderer = {
					toneMapping: 0,
					getRenderTarget: () => null,
					getClearAlpha: () => 1,
					getClearColor() {},
					setClearColor() {},
					setRenderTarget() {},
					render() {},
					readRenderTargetPixelsAsync: async () => new Uint8Array( 4 )
				};

				await assert.rejects(
					renderAndReadTeacher( renderer, {}, {}, {}, 1, 1 ),
					/Half-float teacher readback is required/,
					'does not silently train from LDR pixels'
				);

			} );

		} );

	} );

} );

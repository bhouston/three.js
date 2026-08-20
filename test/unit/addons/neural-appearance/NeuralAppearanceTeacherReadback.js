import { describe, test, expect } from 'vitest';
import {
	readSamplePixel,
	renderAndReadTeacher
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherReadback.js';

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceTeacherReadback', () => {

			test( 'reads the center teacher pixel of a tile', () => {

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

				expect( Number.isFinite( point[ 0 ] ) ).toBeTruthy();
				expect( Number.isFinite( point[ 1 ] ) ).toBeTruthy();
				expect( point[ 2 ] ).toBe( 0 );

			} );

			test( 'rejects non-half-float teacher readback', async () => {

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

				await expect( renderAndReadTeacher( renderer, {}, {}, {}, 1, 1 ) ).rejects.toThrow(
					/Half-float teacher readback is required/
				);

			} );

		} );

	} );

} );

import { describe, expect, it, vi } from 'vitest';
import {
	readPixelValue,
	readSamplePixel,
	renderAndReadTeacher,
	renderAndReadTeacherAttachments
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherReadback.js';

// `validateHalfFloatPixels` is not exported, so it is exercised indirectly
// through `renderAndReadTeacher`/`renderAndReadTeacherAttachments` below
// (the "throws" tests), matching what the module actually makes public.

// Half-float (binary16) bit patterns used to exercise `readPixelValue`'s
// Uint16Array branch against `THREE.DataUtils.fromHalfFloat` without
// depending on that function's own internals: 0x3C00 = 1.0, 0x0000 = 0.0,
// 0xC000 = -2.0 (sign bit set, exponent 16, no separate table needed since
// these are exact powers of two / zero, easy to hand-verify).
const HALF_ONE = 0x3C00;
const HALF_ZERO = 0x0000;
const HALF_NEG_TWO = 0xC000;

function createMockRenderer( { pixels = new Uint16Array( [ HALF_ONE, HALF_ZERO, HALF_ONE, HALF_ONE ] ), renderThrows = false } = {} ) {

	const clearColor = { r: 1, g: 1, b: 1 };

	return {
		_state: {
			renderTarget: 'previous-target',
			toneMapping: 'previous-tone-mapping',
			clearAlpha: 0.25
		},
		getRenderTarget: vi.fn( function () {

			return this._state.renderTarget;

		} ),
		toneMapping: 'previous-tone-mapping',
		getClearColor: vi.fn( ( target ) => {

			target.r = clearColor.r;
			target.g = clearColor.g;
			target.b = clearColor.b;

		} ),
		getClearAlpha: vi.fn( function () {

			return this._state.clearAlpha;

		} ),
		setClearColor: vi.fn(),
		setRenderTarget: vi.fn( function ( target ) {

			this._state.renderTarget = target;

		} ),
		render: vi.fn( () => {

			if ( renderThrows ) throw new Error( 'render failed' );

		} ),
		readRenderTargetPixelsAsync: vi.fn( async () => pixels )
	};

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceTeacherReadback', () => {

	describe( 'readPixelValue', () => {

		it( 'reads a plain array/typed array (non-Uint16Array) value directly, with no decoding', () => {

			const pixels = new Float32Array( [ 0.25, 0.5, 0.75 ] );

			expect( readPixelValue( pixels, 0 ) ).toBe( 0.25 );
			expect( readPixelValue( pixels, 2 ) ).toBe( 0.75 );

		} );

		it( 'reads a plain JS array value directly', () => {

			expect( readPixelValue( [ 1, 2, 3 ], 1 ) ).toBe( 2 );

		} );

		it( 'decodes a Uint16Array value as half-float', () => {

			const pixels = new Uint16Array( [ HALF_ZERO, HALF_ONE, HALF_NEG_TWO ] );

			expect( readPixelValue( pixels, 0 ) ).toBe( 0 );
			expect( readPixelValue( pixels, 1 ) ).toBe( 1 );
			expect( readPixelValue( pixels, 2 ) ).toBe( -2 );

		} );

	} );

	describe( 'readSamplePixel', () => {

		it( 'reads the center texel of tile 0 (top-left tile) as [r,g,b,a]', () => {

			// atlasColumns = 2, tileSize = 4 -> tile 0 spans x:[0,4), y:[0,4),
			// center = (2, 2). atlasWidth = 2*4 = 8.
			// index = (2 * 8 + 2) * 4 = 72
			const atlasColumns = 2;
			const atlasWidth = 8;
			const tileSize = 4;
			const pixels = new Float32Array( atlasWidth * atlasWidth * 4 );
			const index = ( 2 * atlasWidth + 2 ) * 4;
			pixels[ index ] = 10;
			pixels[ index + 1 ] = 20;
			pixels[ index + 2 ] = 30;
			pixels[ index + 3 ] = 40;

			expect( readSamplePixel( pixels, 0, atlasColumns, atlasWidth, tileSize ) ).toEqual( [ 10, 20, 30, 40 ] );

		} );

		it( 'wraps to the next row of tiles once sampleIndex exceeds atlasColumns', () => {

			// atlasColumns = 2, tileSize = 4, sampleIndex = 2 -> tileX = 0, tileY = 1
			// center = (2, 4 + 2) = (2, 6). atlasWidth = 8.
			// index = (6 * 8 + 2) * 4 = 200
			const atlasColumns = 2;
			const atlasWidth = 8;
			const tileSize = 4;
			const pixels = new Float32Array( atlasWidth * atlasWidth * 4 );
			const index = ( 6 * atlasWidth + 2 ) * 4;
			pixels[ index ] = 1;
			pixels[ index + 1 ] = 2;
			pixels[ index + 2 ] = 3;
			pixels[ index + 3 ] = 4;

			expect( readSamplePixel( pixels, 2, atlasColumns, atlasWidth, tileSize ) ).toEqual( [ 1, 2, 3, 4 ] );

		} );

		it( 'floors an odd tileSize center offset (tileSize / 2 is not an integer)', () => {

			// tileSize = 3 -> Math.floor(3/2) = 1, so center offset within the
			// tile is (1, 1), not (1.5, 1.5).
			const atlasColumns = 3;
			const atlasWidth = 9; // 3 columns * tileSize 3
			const tileSize = 3;
			const pixels = new Float32Array( atlasWidth * atlasWidth * 4 );
			// tile 4 -> tileX = 1, tileY = 1 -> x = 3 + 1 = 4, y = 3 + 1 = 4
			const index = ( 4 * atlasWidth + 4 ) * 4;
			pixels[ index ] = 5;
			pixels[ index + 1 ] = 6;
			pixels[ index + 2 ] = 7;
			pixels[ index + 3 ] = 8;

			expect( readSamplePixel( pixels, 4, atlasColumns, atlasWidth, tileSize ) ).toEqual( [ 5, 6, 7, 8 ] );

		} );

		it( 'decodes half-float pixel data through the same tile-coordinate math', () => {

			const atlasColumns = 1;
			const atlasWidth = 4;
			const tileSize = 4;
			const pixels = new Uint16Array( atlasWidth * atlasWidth * 4 );
			const index = ( 2 * atlasWidth + 2 ) * 4;
			pixels[ index ] = HALF_ONE;
			pixels[ index + 1 ] = HALF_ZERO;
			pixels[ index + 2 ] = HALF_NEG_TWO;
			pixels[ index + 3 ] = HALF_ONE;

			expect( readSamplePixel( pixels, 0, atlasColumns, atlasWidth, tileSize ) ).toEqual( [ 1, 0, -2, 1 ] );

		} );

	} );

	describe( 'renderAndReadTeacher / renderAndReadTeacherAttachments (mocked renderer)', () => {

		it( 'renders once and reads back a single attachment (default textureIndices = [0])', async () => {

			const pixels = new Uint16Array( [ HALF_ONE, HALF_ZERO, HALF_ONE, HALF_ONE ] );
			const renderer = createMockRenderer( { pixels } );
			const scene = {};
			const camera = {};
			const target = {};

			const result = await renderAndReadTeacher( renderer, scene, camera, target, 8, 8 );

			expect( renderer.render ).toHaveBeenCalledTimes( 1 );
			expect( renderer.render ).toHaveBeenCalledWith( scene, camera );
			expect( renderer.readRenderTargetPixelsAsync ).toHaveBeenCalledTimes( 1 );
			expect( renderer.readRenderTargetPixelsAsync ).toHaveBeenCalledWith( target, 0, 0, 8, 8, 0 );
			expect( result ).toBe( pixels );

		} );

		it( 'reads back multiple MRT attachments in one render pass, in the requested order', async () => {

			const renderer = createMockRenderer();
			const target = {};

			const result = await renderAndReadTeacherAttachments( renderer, {}, {}, target, 4, 4, [ 2, 0, 1 ] );

			expect( renderer.render ).toHaveBeenCalledTimes( 1 ); // one render, not one per attachment
			expect( renderer.readRenderTargetPixelsAsync ).toHaveBeenCalledTimes( 3 );
			expect( renderer.readRenderTargetPixelsAsync ).toHaveBeenNthCalledWith( 1, target, 0, 0, 4, 4, 2 );
			expect( renderer.readRenderTargetPixelsAsync ).toHaveBeenNthCalledWith( 2, target, 0, 0, 4, 4, 0 );
			expect( renderer.readRenderTargetPixelsAsync ).toHaveBeenNthCalledWith( 3, target, 0, 0, 4, 4, 1 );
			expect( result ).toHaveLength( 3 );

		} );

		it( 'sets render state (target, tone mapping, clear color/alpha) before rendering', async () => {

			const renderer = createMockRenderer();
			let toneMappingDuringRender;
			renderer.render.mockImplementation( () => {

				toneMappingDuringRender = renderer.toneMapping;

			} );

			await renderAndReadTeacher( renderer, {}, {}, 'the-target', 4, 4 );

			expect( renderer.setRenderTarget ).toHaveBeenCalledWith( 'the-target' );
			expect( renderer.setClearColor ).toHaveBeenCalledWith( 0x000000, 1 );
			expect( toneMappingDuringRender ).toBe( 0 ); // THREE.NoToneMapping === 0, restored by the time render() returns

		} );

		it( 'restores the previous render target, tone mapping, and clear color/alpha after a successful render', async () => {

			const renderer = createMockRenderer();

			await renderAndReadTeacher( renderer, {}, {}, {}, 4, 4 );

			expect( renderer.setRenderTarget ).toHaveBeenLastCalledWith( 'previous-target' );
			expect( renderer.toneMapping ).toBe( 'previous-tone-mapping' );
			expect( renderer.setClearColor ).toHaveBeenLastCalledWith(
				expect.objectContaining( { r: 1, g: 1, b: 1 } ),
				0.25
			);

		} );

		it( 'restores render state even when renderer.render() throws (finally block)', async () => {

			const renderer = createMockRenderer( { renderThrows: true } );

			await expect( renderAndReadTeacher( renderer, {}, {}, {}, 4, 4 ) ).rejects.toThrow( 'render failed' );

			expect( renderer.setRenderTarget ).toHaveBeenLastCalledWith( 'previous-target' );
			expect( renderer.toneMapping ).toBe( 'previous-tone-mapping' );
			expect( renderer.readRenderTargetPixelsAsync ).not.toHaveBeenCalled();

		} );

		it( 'throws when the readback pixels are not a Uint16Array (half-float required), and still restores state', async () => {

			const renderer = createMockRenderer( { pixels: new Float32Array( [ 1, 2, 3, 4 ] ) } );

			await expect( renderAndReadTeacher( renderer, {}, {}, {}, 4, 4 ) ).rejects.toThrow(
				'Half-float teacher readback is required; received Float32Array.'
			);

			expect( renderer.setRenderTarget ).toHaveBeenLastCalledWith( 'previous-target' );
			expect( renderer.toneMapping ).toBe( 'previous-tone-mapping' );

		} );

		it( 'reports the constructor name of a non-typed-array readback value in the error message', async () => {

			const renderer = createMockRenderer( { pixels: [ 1, 2, 3, 4 ] } ); // plain Array, not Uint16Array

			await expect( renderAndReadTeacherAttachments( renderer, {}, {}, {}, 4, 4, [ 0 ] ) ).rejects.toThrow(
				'Half-float teacher readback is required; received Array.'
			);

		} );

	} );

} );

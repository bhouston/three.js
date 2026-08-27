import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { uv, vec4 } from 'three/tsl';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// This is a general three.js core regression test (src/renderers/webgpu/
// utils/WebGPUTextureUtils.js's copyTextureToBuffer, used by both
// Renderer.readRenderTargetPixelsAsync and, indirectly, getArrayBufferAsync)
// - not specific to the NTC addon. It lives under this vitest project purely
// because this is the one already configured with real WebGPU launch args
// (see vitest.config.js) - the bug it guards against was originally found
// (and worked around, at several call sites) while developing the NTC
// addon, documented in test/vitest/browser/ntc/NTCDecoderTSL.test.js and
// NTCPackedDotProduct.test.js.
//
// WebGPU's copyTextureToBuffer requires GPUImageCopyBuffer.bytesPerRow to be
// a multiple of 256 (COPY_BYTES_PER_ROW_ALIGNMENT). For a texture/render
// target whose *true* per-row byte size is smaller than that (e.g. an
// RGBA16F target narrower than 32 texels: 32 * 4 channels * 2 bytes = 256
// exactly, so 32 is the smallest width needing no padding at all), the GPU
// pads each copied row out to that 256-byte stride - real bytes for `width *
// bytesPerTexel` bytes, followed by unused padding bytes, repeated per row.
// `copyTextureToBuffer` already computes a buffer sized to hold this padded
// layout correctly, but returns the raw padded ArrayBuffer's bytes directly,
// with no step to strip the inter-row padding back out - so every caller,
// which assumes a tightly-packed `width * height` array of texel data,
// silently reads row 1 onward starting at the wrong offset (row 0 alone is
// unaffected, since it starts at byte 0 either way).
//
// Deliberately convention-agnostic about *which* screen row a given `v`
// value ends up at (render-target vs. texture-sample Y orientation is a
// real, separate, already-understood three.js convention - see
// NTCTextureSource.js's bakeColorNodeToTexture doc comment on PlaneGeometry's
// default UV needing a deliberate V-flip to round-trip correctly - not the
// bug this file targets). Every check here only asks "is a smooth, evenly-
// stepped gradient actually present at every row/column", which the padding
// bug breaks (rows read back as duplicated/zeroed garbage) regardless of
// which physical direction increasing `v` happens to render into.
describe( 'Renderer.readRenderTargetPixelsAsync - RGBA16F row-alignment (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	/**
	 * Renders `uv()` (red = u, green = v) to a `size`x`size` RGBA16F render
	 * target and reads every pixel back, returning a `[row][col] -> [r,g,b,a]`
	 * grid of decoded floats.
	 */
	async function renderUvGrid( size ) {

		const scene = new THREE.Scene();
		const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
		camera.position.set( 0, 0, 2 );

		const material = new THREE.NodeMaterial();
		material.lights = false;
		material.toneMapped = false;
		material.fragmentNode = vec4( uv(), 0, 1 );

		const mesh = new THREE.Mesh( new THREE.PlaneGeometry( 2, 2 ), material );
		scene.add( mesh );

		const renderTarget = new THREE.RenderTarget( size, size, { type: THREE.HalfFloatType } );
		renderer.setRenderTarget( renderTarget );
		renderer.render( scene, camera );
		renderer.setRenderTarget( null );

		const pixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, size, size );

		renderTarget.dispose();
		material.dispose();
		mesh.geometry.dispose();

		const grid = [];
		for ( let row = 0; row < size; row ++ ) {

			const rowValues = [];
			for ( let col = 0; col < size; col ++ ) {

				const i = ( row * size + col ) * 4;
				rowValues.push( [ 0, 1, 2, 3 ].map( ( c ) => THREE.DataUtils.fromHalfFloat( pixels[ i + c ] ) ) );

			}

			grid.push( rowValues );

		}

		return grid;

	}

	/**
	 * Asserts that `values` (in column/row scan order) forms an evenly-
	 * stepped gradient covering the full `[0.5/size, (size-0.5)/size]` range
	 * with step `1/size` - i.e. a genuine, undamaged per-texel gradient, not
	 * a run of duplicated/zeroed garbage. Sorted first, since this file makes
	 * no assumption about *which* screen direction increasing u/v renders
	 * into (see this file's own doc comment).
	 */
	function expectEvenGradient( values, size ) {

		const sorted = [ ...values ].sort( ( a, b ) => a - b );
		const step = 1 / size;

		for ( let i = 0; i < size; i ++ ) {

			const expected = ( i + 0.5 ) * step;
			expect( sorted[ i ] ).toBeCloseTo( expected, 2 );

		}

		// A genuine gradient has `size` *distinct* values - the padding bug's
		// failure mode (rows collapsing to duplicated/zeroed data) would
		// instead show far fewer distinct values than `size`.
		const distinctCount = new Set( sorted.map( ( v ) => Math.round( v * 4096 ) ) ).size;
		expect( distinctCount ).toBe( size );

	}

	// 32 texels is the smallest RGBA16F width whose natural row byte size
	// (32 * 4 * 2 = 256) already meets WebGPU's 256-byte row-alignment
	// requirement with zero padding - i.e. the smallest size this bug cannot
	// possibly affect. Used as the known-good baseline.
	it( 'a 32-texel-wide target (no row padding needed) reads back a clean gradient at every row and column', async () => {

		const size = 32;
		const grid = await renderUvGrid( size );

		// Every row's green (v) channel should be constant across that row,
		// and the size distinct row-constants should form a clean gradient.
		const rowGreens = grid.map( ( row ) => row[ 0 ][ 1 ] );
		expectEvenGradient( rowGreens, size );

		// Every column's red (u) channel should be constant down that column
		// - checked via the *last* row specifically (not row 0), since row 0
		// alone starting at buffer offset 0 is exactly what the bug this file
		// targets does NOT corrupt.
		const lastRowReds = grid[ size - 1 ].map( ( pixel ) => pixel[ 0 ] );
		expectEvenGradient( lastRowReds, size );

	} );

	// 8 texels: natural row byte size is 8 * 4 * 2 = 64 bytes, well under the
	// 256-byte alignment boundary - every row but the first needs padding
	// stripped. This is the exact size class that silently produced
	// all-zero texels during NTC development (see NTCPackedDotProduct.test.js's
	// plan-doc-referenced history) before this was understood to be a core
	// bug rather than an addon issue.
	it( 'an 8-texel-wide target (real row padding required) reads back a clean gradient at every row and column, not just row 0', async () => {

		const size = 8;
		const grid = await renderUvGrid( size );

		const rowGreens = grid.map( ( row ) => row[ 0 ][ 1 ] );
		expectEvenGradient( rowGreens, size );

		const lastRowReds = grid[ size - 1 ].map( ( pixel ) => pixel[ 0 ] );
		expectEvenGradient( lastRowReds, size );

	} );

	it( 'a 1-texel-wide target (maximally padded: 8 real bytes in a 256-byte row) still reads back correctly', async () => {

		const grid = await renderUvGrid( 1 );

		expect( grid[ 0 ][ 0 ][ 0 ] ).toBeCloseTo( 0.5, 2 );
		expect( grid[ 0 ][ 0 ][ 1 ] ).toBeCloseTo( 0.5, 2 );

	} );

} );

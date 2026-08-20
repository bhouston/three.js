import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { vec4 } from 'three/tsl';
import {
	computeAtlasDimensions,
	createTeacherRenderTarget,
	createAtlasTextures,
	createAtlasShaderNodes,
	uploadAtlasSamples
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTeacherAtlas.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// bakeColorNodeToTexture (NeuralTextureSource.js)/NeuralAppearanceTeacherEvaluator.js's
// own render targets are HalfFloatType, so readback pixels arrive as raw
// half-float bit patterns - see NeuralAppearanceTeacherReadback.js's
// readPixelValue for the same convention used in production.
function readHalfFloatPixels( pixels ) {

	const out = new Float32Array( pixels.length );
	for ( let i = 0; i < pixels.length; i ++ ) out[ i ] = THREE.DataUtils.fromHalfFloat( pixels[ i ] );
	return out;

}

// Renders a single unlit full-screen node (one of createAtlasShaderNodes'
// outputs, or any TSL expression built from it) into a real atlas render
// target and reads every pixel back, exactly mirroring the scene/camera/
// material shape NeuralAppearanceTeacherEvaluator._createResources uses for
// the real teacher render, minus lighting (not needed here - these nodes are
// pure geometry/sample decode, not BRDF evaluation).
async function renderAtlasNode( renderer, node, atlasWidth, atlasHeight ) {

	const target = createTeacherRenderTarget( atlasWidth, atlasHeight );

	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
	camera.position.set( 0, 0, 2 );

	const material = new THREE.NodeMaterial();
	material.lights = false;
	material.toneMapped = false;
	material.blending = THREE.NoBlending;
	// `fragmentNode`, not `colorNode`, to bypass setupDiffuseColor's unsigned
	// clamp - see NeuralTextureSource.js's bakeColorNodeToTexture for the
	// same reasoning (this node graph can and does carry signed data, e.g.
	// normal components in [-1, 1]).
	material.fragmentNode = vec4( node );

	const mesh = new THREE.Mesh( new THREE.PlaneGeometry( 2, 2 ), material );
	scene.add( mesh );

	const previousTarget = renderer.getRenderTarget();
	renderer.setRenderTarget( target );
	renderer.render( scene, camera );
	const rawPixels = await renderer.readRenderTargetPixelsAsync( target, 0, 0, atlasWidth, atlasHeight, 0 );
	renderer.setRenderTarget( previousTarget );

	const pixels = readHalfFloatPixels( rawPixels );

	target.dispose();
	material.dispose();
	mesh.geometry.dispose();

	return pixels;

}

function pixelAt( pixels, atlasWidth, x, y ) {

	const index = ( y * atlasWidth + x ) * 4;
	return [ pixels[ index ], pixels[ index + 1 ], pixels[ index + 2 ], pixels[ index + 3 ] ];

}

// Same "tile index -> pixel" mapping NeuralAppearanceTeacherReadback.js's
// readSamplePixel uses in production to read a rendered atlas back out -
// deliberately re-derived here from the tile-grid definition itself (column-
// major index, tile center = tile origin + half the tile size) rather than
// imported, so a disagreement between how NeuralAppearanceTeacherAtlas.js
// *writes* samples into tiles and how the real consumer *reads* them back
// would show up as a genuine test failure instead of being masked by reusing
// the same helper on both sides.
function tileCenterPixel( sampleIndex, atlasColumns, tileSize ) {

	const tileX = sampleIndex % atlasColumns;
	const tileY = Math.floor( sampleIndex / atlasColumns );
	return {
		x: tileX * tileSize + Math.floor( tileSize / 2 ),
		y: tileY * tileSize + Math.floor( tileSize / 2 )
	};

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceTeacherAtlas', () => {

	// computeAtlasDimensions is pure arithmetic (no GPU involved) - hand
	// computed expectations below, independent of the module's own formula.
	describe( 'computeAtlasDimensions (pure math)', () => {

		it( 'packs a capacity of 1 into a single, 16-aligned tile column/row', () => {

			// sqrt(1) = 1, aligned up to the default alignment of 16 -> 16
			// columns, 1 row (ceil(1/16) = 1).
			const dims = computeAtlasDimensions( 1, 8 );

			expect( dims.atlasColumns ).toBe( 16 );
			expect( dims.atlasRows ).toBe( 1 );
			expect( dims.atlasWidth ).toBe( 16 * 8 );
			expect( dims.atlasHeight ).toBe( 8 );

		} );

		it( 'packs a capacity of 300 into a roughly square, 16-aligned grid', () => {

			// sqrt(300) ~= 17.32 -> ceil = 18 -> aligned up to the next
			// multiple of 16 -> 32 columns. rows = ceil(300 / 32) = 10.
			const dims = computeAtlasDimensions( 300, 8 );

			expect( dims.atlasColumns ).toBe( 32 );
			expect( dims.atlasRows ).toBe( 10 );
			expect( dims.atlasWidth ).toBe( 32 * 8 );
			expect( dims.atlasHeight ).toBe( 10 * 8 );
			// Every one of the 300 requested tiles must actually fit in the
			// grid - the one property any packing scheme must satisfy.
			expect( dims.atlasColumns * dims.atlasRows ).toBeGreaterThanOrEqual( 300 );

		} );

		it( 'treats capacity 0 the same as capacity 1 (degenerate/empty batch, still a valid non-zero-area atlas)', () => {

			const dims = computeAtlasDimensions( 0, 8 );

			expect( dims.atlasWidth ).toBeGreaterThan( 0 );
			expect( dims.atlasHeight ).toBeGreaterThan( 0 );

		} );

	} );

	describe( 'rendered atlas content (real WebGPU)', () => {

		let renderer;

		beforeAll( async () => {

			renderer = await createTestRenderer();

		} );

		afterAll( () => {

			renderer?.dispose();
			renderer = undefined;

		} );

		// A generously-sized grid (128x144 px) - small enough to render
		// instantly, comfortably clear of the tiny-render-target readback
		// quirk documented in NeuralTextureSource.test.js (that quirk was
		// observed well under 32px on a side).
		const atlasColumns = 8;
		const atlasRows = 9;
		const tileSize = 16;
		const atlasWidth = atlasColumns * tileSize;
		const atlasHeight = atlasRows * tileSize;

		it( 'decodes each uploaded sample normal to its known unit vector, at the correct, non-overlapping tile', () => {

			return ( async () => {

				const sampleTextures = createAtlasTextures( atlasColumns, atlasRows );
				const nodes = createAtlasShaderNodes( sampleTextures, atlasColumns, atlasRows, tileSize );

				// Hand-picked non-unit inputs with independently-known
				// normalized results (Pythagorean triples / sqrt(3)), placed
				// at tile indices spanning two different rows (index 8 wraps
				// to row 1 with atlasColumns=8) so a row/column transposition
				// or vertical-flip bug would also be caught.
				const samples = [
					{ normal: [ 3, 4, 0 ] },  // |v|=5  -> [0.6, 0.8, 0]
					{ normal: [ 0, 0, -5 ] }, // |v|=5  -> [0, 0, -1]
					{ normal: [ 1, 1, 1 ] },  // |v|=sqrt(3) -> [1,1,1]/sqrt(3)
					{ normal: [ 0, -3, 4 ] }  // -> [0, -0.6, 0.8]
				];
				// Placed at indices 0, 1, 2 (row 0) and 8 (row 1, column 0).
				const sparseSamples = [ samples[ 0 ], samples[ 1 ], samples[ 2 ], {}, {}, {}, {}, {}, samples[ 3 ] ];

				uploadAtlasSamples( sampleTextures, sparseSamples, atlasColumns, atlasHeight, tileSize );

				const pixels = await renderAtlasNode( renderer, nodes.normal, atlasWidth, atlasHeight );

				const invSqrt3 = 1 / Math.sqrt( 3 );
				const expectations = [
					{ index: 0, value: [ 0.6, 0.8, 0 ] },
					{ index: 1, value: [ 0, 0, - 1 ] },
					{ index: 2, value: [ invSqrt3, invSqrt3, invSqrt3 ] },
					{ index: 8, value: [ 0, - 0.6, 0.8 ] }
				];

				for ( const { index, value } of expectations ) {

					const { x, y } = tileCenterPixel( index, atlasColumns, tileSize );
					const [ r, g, b ] = pixelAt( pixels, atlasWidth, x, y );

					expect( r ).toBeCloseTo( value[ 0 ], 3 );
					expect( g ).toBeCloseTo( value[ 1 ], 3 );
					expect( b ).toBeCloseTo( value[ 2 ], 3 );

				}

			} )();

		} );

		it( 'reconstructs an unchanged material UV across an entire tile when the uploaded UV gradient is zero (identity case)', () => {

			return ( async () => {

				const sampleTextures = createAtlasTextures( atlasColumns, atlasRows );
				const nodes = createAtlasShaderNodes( sampleTextures, atlasColumns, atlasRows, tileSize );

				const uv = [ 0.5, 0.5 ];
				const samples = [ { uv, duvDx: [ 0, 0 ], duvDy: [ 0, 0 ] } ];

				uploadAtlasSamples( sampleTextures, samples, atlasColumns, atlasHeight, tileSize );

				const pixels = await renderAtlasNode( renderer, vec4( nodes.materialUv, 0, 1 ), atlasWidth, atlasHeight );

				// Sample 3 different positions inside tile 0: its center, and
				// two corners. With zero gradient, every position within the
				// tile must reconstruct exactly the uploaded UV - a plain
				// identity-transform property, independent of any offset math.
				const positions = [
					{ x: Math.floor( tileSize / 2 ), y: Math.floor( tileSize / 2 ) },
					{ x: 0, y: 0 },
					{ x: tileSize - 1, y: tileSize - 1 }
				];

				for ( const { x, y } of positions ) {

					const [ u, v ] = pixelAt( pixels, atlasWidth, x, y );
					expect( u ).toBeCloseTo( uv[ 0 ], 2 );
					expect( v ).toBeCloseTo( uv[ 1 ], 2 );

				}

			} )();

		} );

		it( 'reconstructs the material UV at the tile-left edge as uv - 0.5*duvDx, a hand-derived value from the uploaded gradient', () => {

			return ( async () => {

				const sampleTextures = createAtlasTextures( atlasColumns, atlasRows );
				const nodes = createAtlasShaderNodes( sampleTextures, atlasColumns, atlasRows, tileSize );

				const uv = [ 0.5, 0.5 ];
				const duvDx = [ 0.2, 0 ];
				const duvDy = [ 0, 0 ];
				const samples = [ { uv, duvDx, duvDy } ];

				uploadAtlasSamples( sampleTextures, samples, atlasColumns, atlasHeight, tileSize );

				const pixels = await renderAtlasNode( renderer, vec4( nodes.materialUv, 0, 1 ), atlasWidth, atlasHeight );

				// Left column of the tile (local integer pixel x=0): GPU
				// fragment shaders sample at the *pixel center*, i.e. local
				// coordinate 0.5, not 0 - a standard rasterization
				// convention, not something specific to this module. So the
				// fractional tile position along x is 0.5/tileSize, and the
				// [-0.5, 0.5)-centered tile-local offset is
				// tilePixel.x = 0.5/tileSize - 0.5. duvDy is zero, so the y
				// term drops out regardless of tilePixel.y.
				const tilePixelX = 0.5 / tileSize - 0.5;
				const expectedU = uv[ 0 ] + duvDx[ 0 ] * tilePixelX;
				const expectedV = uv[ 1 ] + duvDx[ 1 ] * tilePixelX;

				const [ u, v ] = pixelAt( pixels, atlasWidth, 0, Math.floor( tileSize / 2 ) );

				expect( u ).toBeCloseTo( expectedU, 2 );
				expect( v ).toBeCloseTo( expectedV, 2 );

			} )();

		} );

	} );

} );

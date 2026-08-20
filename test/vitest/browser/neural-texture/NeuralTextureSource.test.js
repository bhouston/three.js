import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { DataUtils } from 'three';
import { uv, vec4 } from 'three/tsl';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/neural-texture/NeuralTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// bakeColorNodeToTexture's render targets are HalfFloatType, so
// readRenderTargetPixelsAsync returns raw half-float bit patterns as a
// Uint16Array, not decoded floats - see NeuralAppearanceTeacherReadback.js's
// readPixelValue for the same pattern this repo already uses elsewhere.
function readHalfFloatPixels( pixels ) {

	const out = new Float32Array( pixels.length );
	for ( let i = 0; i < pixels.length; i ++ ) out[ i ] = DataUtils.fromHalfFloat( pixels[ i ] );
	return out;

}

// bakeColorNodeToTexture is what turns a MaterialX node graph (e.g. a
// tangent-space (dx, dy) normal offset, signed in [-1, 1]) into the training
// ground truth NeuralTextureTrainer samples from. Two ways this could quietly
// produce a training target with the wrong sign (which would look exactly
// like "the network never learns negative-going bumps", one of the leading
// hypotheses for the neural-material "brick" bug): the render target
// silently clamping negative values to 0, or the documented V-flip changing
// which physical UV corner a given pixel/sample actually represents.
describe( 'Addons > Neural > NeuralTexture > NeuralTextureSource.bakeColorNodeToTexture (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	it( 'preserves negative values (HalfFloat target, no [0,1] clamp)', async () => {

		// Signed ramps across the quad, exactly the shape a real (dx, dy) normal
		// offset target has: [-1, 1] in R (from raw u) and [-1, 1] in G (from
		// raw v).
		const colorNode = vec4( uv().x.mul( 2 ).sub( 1 ), uv().y.mul( 2 ).sub( 1 ), 0, 1 );

		// Real bake resolutions are always >= 128 (see the example's
		// `bakeResolution` GUI options); a tiny resolution like 4 or 8 here hits
		// an unrelated readRenderTargetPixelsAsync quirk (rows appear to only be
		// partially populated) that has nothing to do with what this test is
		// checking - 32 is comfortably clear of that.
		const resolution = 32;
		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, resolution );

		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, resolution, resolution ) );

		let sawNegativeR = false;
		let sawPositiveR = false;
		let sawNegativeG = false;
		let sawPositiveG = false;

		for ( let i = 0; i < resolution * resolution; i ++ ) {

			const r = pixels[ i * 4 + 0 ];
			const g = pixels[ i * 4 + 1 ];

			if ( r < - 0.01 ) sawNegativeR = true;
			if ( r > 0.01 ) sawPositiveR = true;
			if ( g < - 0.01 ) sawNegativeG = true;
			if ( g > 0.01 ) sawPositiveG = true;

		}

		expect( sawNegativeR ).toBe( true );
		expect( sawPositiveR ).toBe( true );
		expect( sawNegativeG ).toBe( true );
		expect( sawPositiveG ).toBe( true );

		renderTarget.dispose();

	} );

	it( 'round-trips raw UV consistently: baking at (u, v) and reading back at pixel (u, v) agree, despite the internal V-flip', async () => {

		// This is the exact contract flattenChannelComponents/the training
		// kernel depend on (see the long comment in NeuralTextureSource.js): the
		// quad's UV is flipped internally before render so that "bake a value
		// at raw (u, v)" and "sample the resulting texture back at raw (u, v)"
		// round-trip correctly - i.e. render-target row 0 (v=0 in texture-
		// sample space) must correspond to whatever was baked at raw v=0, not
		// raw v=1.
		// See the resolution comment in the previous test - stay well clear of
		// the tiny-resolution readback quirk.
		const resolution = 32;
		// A pattern that's asymmetric in both axes so a flip, transpose, or
		// rotation of the readback all produce a detectably different result:
		// R ramps 0->1 with u, G ramps 0->1 with v, B is a single corner marker
		// only nonzero very close to (u=0, v=0).
		const colorNode = vec4(
			uv().x,
			uv().y,
			uv().x.lessThan( 0.1 ).and( uv().y.lessThan( 0.1 ) ).select( 1, 0 ),
			1
		);

		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, resolution );
		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, resolution, resolution ) );

		// texture()/textureLevel() sample (u, v) with v=0 at *row 0* of the
		// backing data (standard GPU texture-sample convention) - so if the bake
		// round-trips raw UV correctly, the corner marker (baked at raw u=0,
		// v=0) must read back at row 0, column 0 of the pixel buffer.
		const cornerIndex = 0; // row 0, col 0
		const b = pixels[ cornerIndex * 4 + 2 ];
		expect( b ).toBeCloseTo( 1, 2 );

		// And the far corner (raw u≈1, v≈1) should read back with r≈1, g≈1 at
		// the last row's last column.
		const farIndex = ( resolution - 1 ) * resolution + ( resolution - 1 );
		const rFar = pixels[ farIndex * 4 + 0 ];
		const gFar = pixels[ farIndex * 4 + 1 ];
		expect( rFar ).toBeGreaterThan( 0.5 );
		expect( gFar ).toBeGreaterThan( 0.5 );

		renderTarget.dispose();

	} );

} );

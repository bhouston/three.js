import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { DataUtils } from 'three';
import { vec2, vec4 } from 'three/tsl';
import { reconstructFinalNormal } from '../../../../examples/jsm/ntc/NTCNodeMaterial.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/NTCTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// reconstructFinalNormal blends a trained tangent-space (dx, dy) offset
// through a real mesh's tangentWorld/bitangentWorld/normalWorld. It's the
// function most directly implicated by the reported "neural-material brick
// preset is missing bump detail on some sides" symptom, so this file's job
// is narrow and specific: does it treat *positive and negative* dx/dy
// symmetrically, or does one sign collapse toward the other (which would
// look exactly like "half the brick edges are missing")?
//
// Rather than hardcoding the exact tangent/bitangent axis directions
// computeTangents() happens to produce for a flat quad (an assumption that
// would make this test fragile and wouldn't actually probe sign fidelity),
// every assertion below is *symmetry-based*: does output(-x) differ from
// output(+x) by roughly the same amount output(+x) differs from the flat
// baseline? That's true regardless of which world axis the tangent/
// bitangent basis happens to be, and false if either sign is being silently
// lost.
//
// reconstructFinalNormal is evaluated here with *constant* (non-spatially-
// varying) (dx, dy) via bakeColorNodeToTexture, reusing the exact same
// flat-quad-plus-orthographic-camera setup (and therefore the exact same
// tangentWorld/bitangentWorld/normalWorld construction) NTCSource
// uses to bake real training targets - so this is testing the function
// against a materially identical basis to production, not a synthetic one.
describe( 'Addons > Neural > NeuralMaterial > NTCNodeMaterial.reconstructFinalNormal (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	async function evaluateNormal( dx, dy ) {

		const colorNode = vec4( reconstructFinalNormal( vec2( dx, dy ) ).mul( 0.5 ).add( 0.5 ), 1 );
		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, 2 );
		const rawPixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, 1, 1 );
		renderTarget.dispose();

		// bakeColorNodeToTexture's render target is HalfFloatType, so the
		// readback is a Uint16Array of raw half-float bit patterns - see
		// NeuralAppearanceTeacherReadback.js's readPixelValue for the same
		// decode step this repo already uses elsewhere.
		const pixels = [ 0, 1, 2 ].map( ( i ) => DataUtils.fromHalfFloat( rawPixels[ i ] ) );

		// Undo the *0.5+0.5 display remap to get back the actual (view-space)
		// normal components, in the same [-1, 1] range dx/dy were given in.
		return [ pixels[ 0 ] * 2 - 1, pixels[ 1 ] * 2 - 1, pixels[ 2 ] * 2 - 1 ];

	}

	function distance( a, b ) {

		return Math.hypot( a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] );

	}

	it( 'the flat (dx=0, dy=0) case is a unit vector (sanity check on the whole pipeline)', async () => {

		const normal = await evaluateNormal( 0, 0 );
		const length = Math.hypot( ...normal );

		expect( length ).toBeCloseTo( 1, 3 );

	} );

	it( 'positive and negative dx produce clearly distinct, non-collapsed results', async () => {

		const baseline = await evaluateNormal( 0, 0 );
		const positive = await evaluateNormal( 0.4, 0 );
		const negative = await evaluateNormal( - 0.4, 0 );

		const dPos = distance( positive, baseline );
		const dNeg = distance( negative, baseline );
		const dBetween = distance( positive, negative );

		// Both signs must actually move the result away from flat...
		expect( dPos ).toBeGreaterThan( 0.05 );
		expect( dNeg ).toBeGreaterThan( 0.05 );
		// ...and by a comparable amount - if one sign were collapsing toward
		// the flat baseline (the "half the tanh range never learned" failure
		// mode), dPos and dNeg would diverge sharply instead of matching.
		expect( Math.abs( dPos - dNeg ) ).toBeLessThan( 0.05 );
		// And the two results must be clearly distinguishable from each other,
		// not just both "different from flat" in a way that happens to land in
		// the same place.
		expect( dBetween ).toBeGreaterThan( 0.05 );

	} );

	it( 'positive and negative dy produce clearly distinct, non-collapsed results', async () => {

		const baseline = await evaluateNormal( 0, 0 );
		const positive = await evaluateNormal( 0, 0.4 );
		const negative = await evaluateNormal( 0, - 0.4 );

		const dPos = distance( positive, baseline );
		const dNeg = distance( negative, baseline );
		const dBetween = distance( positive, negative );

		expect( dPos ).toBeGreaterThan( 0.05 );
		expect( dNeg ).toBeGreaterThan( 0.05 );
		expect( Math.abs( dPos - dNeg ) ).toBeLessThan( 0.05 );
		expect( dBetween ).toBeGreaterThan( 0.05 );

	} );

	it( 'all four sign combinations of (dx, dy) - the four "brick edge" cases - are mutually distinguishable', async () => {

		// This is the direct analogue of "does every one of the four brick
		// edges (each driven by one sign combination of (dx, dy)) actually
		// produce a different-looking normal" - if reconstructFinalNormal
		// itself were the bug, two of these four corners would land on top of
		// each other (or on the flat baseline).
		const corners = {
			'(+,+)': await evaluateNormal( 0.4, 0.4 ),
			'(+,-)': await evaluateNormal( 0.4, - 0.4 ),
			'(-,+)': await evaluateNormal( - 0.4, 0.4 ),
			'(-,-)': await evaluateNormal( - 0.4, - 0.4 )
		};

		const keys = Object.keys( corners );

		for ( let i = 0; i < keys.length; i ++ ) {

			for ( let j = i + 1; j < keys.length; j ++ ) {

				const d = distance( corners[ keys[ i ] ], corners[ keys[ j ] ] );
				expect( d, `${keys[ i ]} vs ${keys[ j ]} should be distinguishable` ).toBeGreaterThan( 0.05 );

			}

		}

	} );

	it( 'dx/dy near/over the unit-circle boundary stay finite (z reconstruction clamps instead of producing NaN)', async () => {

		// tanh individually bounds dx and dy to (-1, 1), but dx*dx + dy*dy can
		// still exceed 1 - exactly the case reconstructFinalNormal's
		// `max(1 - dx*dx - dy*dy, 0)` guards against.
		const overBoundary = await evaluateNormal( 0.9, 0.9 ); // dx^2+dy^2 = 1.62
		const onBoundary = await evaluateNormal( 0.6, 0.8 ); // dx^2+dy^2 = 1.0 exactly

		for ( const component of [ ...overBoundary, ...onBoundary ] ) {

			expect( Number.isFinite( component ) ).toBe( true );

		}

		expect( Math.hypot( ...overBoundary ) ).toBeCloseTo( 1, 2 );
		expect( Math.hypot( ...onBoundary ) ).toBeCloseTo( 1, 2 );

	} );

} );

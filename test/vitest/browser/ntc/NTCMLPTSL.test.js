import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { uv, float, vec4 } from 'three/tsl';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';
import { forwardMLP } from '../../../../examples/jsm/ntc/training/NTCMLP.js';
import { evaluateNeuralTextureRaw, buildMipChainTexture } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// Regression coverage for a real bug: NTCMLPTSL.js's `hardGeluTSL` (the
// vec4/hvec4-packed twin evaluateLinearLayerMat4 uses at inference time, one
// mat4 block = 4 neurons at once) used to be written with `select()`:
//
//   const upper = select( x.greaterThanEqual( 1.5 ), x, middle );
//   return select( x.lessThanEqual( -1.5 ), zero, upper );
//
// three.js's `select()` (ConditionalNode) always narrows its condition to a
// single scalar `bool` before branching - even when the condition was itself
// a per-component `bvec4` comparison on a vec4 `x` - so it picks ONE branch
// for the whole 4-wide vector rather than selecting component-wise. That is
// invisible whenever every one of the 4 packed neurons' pre-activations
// happens to land on the same side of the same breakpoint (e.g. random/small
// initial weights, which is all NTCDecoderTSL.test.js's existing hgelu
// coverage exercises - see its "matches the CPU forward pass when the hidden
// layer uses hgelu" test) - but once real training pushes some (not all) of
// the 4 lanes in a packed group past +-1.5, e.g. to represent sharp,
// high-contrast spatial detail, it silently mis-selects the wrong lanes and
// blows up (the unbounded-outside-its-domain quadratic middle branch). This
// exact test - hand-built pre-activations straddling +-1.5 within a single
// packed vec4 group - reproduced a CPU-vs-GPU divergence of MAE ~110, max
// abs ~1100 pre-fix, and passes tightly now that `select()` (ConditionalNode)
// itself selects per-component for vector conditions instead of narrowing
// them to a single scalar bool; `hardGeluTSL` is back to its original
// `select()`-based form (see its doc comment).
//
// Deliberately hand-constructs decoder weights (rather than actually
// training) so the straddling pre-activations are exact and deterministic,
// and the test stays fast - training convergence is not what's being
// checked here (NTCTrainer.convergence.test.js already covers that).

function readHalfFloatPixels( pixels ) {

	const out = new Float32Array( pixels.length );
	for ( let i = 0; i < pixels.length; i ++ ) out[ i ] = THREE.DataUtils.fromHalfFloat( pixels[ i ] );
	return out;

}

async function renderRawOutputs( renderer, cpuModel, gridSize ) {

	const mipChainTexture = buildMipChainTexture( cpuModel );
	// `renderer` (not `null`) so this exercises whichever storage precision
	// (fp16 `hmat4`/`hvec4`, or the fp32 uniformArray fallback) the real
	// renderer actually picks - the original bug reproduced identically in
	// both, so this test doesn't need to force one or the other.
	const raw = evaluateNeuralTextureRaw( uv(), cpuModel, mipChainTexture, renderer, float( 0 ) );
	const colorNode = vec4( raw[ 0 ], raw[ 1 ], raw[ 2 ], 0 );

	const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, gridSize );
	const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );

	renderTarget.dispose();
	mipChainTexture.dispose();

	return pixels;

}

describe( 'Addons > NTC > NTCMLPTSL hardGeluTSL vec4 packing (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	it( 'matches the CPU forward pass when one packed vec4 group of hidden neurons straddles both +-1.5 hardGELU breakpoints', async () => {

		// >= 32 texels wide - a render target narrower than that has an
		// RGBA16F row byte size below 256, which has been observed to hit a
		// WebGPU pixel-readback row-alignment bug unrelated to this addon or
		// hgelu (see NTCDecoderTSL.test.js's "selects the correct stored grid
		// level" test for the same note) - not something to work around here,
		// just avoided by testing at a safely-aligned size.
		const gridSize = 32;
		// channels: 3 -> decoder input size channels+1 = 4, exactly one packed
		// vec4 group, no zero-padding to obscure the effect. hiddenSizes: [4]
		// -> exactly one packed vec4 group of hidden neurons too.
		const options = { channels: 3, levels: 1, baseResolution: gridSize, hiddenSizes: [ 4 ], outputChannels: 3, hiddenActivation: 'hgelu' };
		const cpuModel = createNTCGridPyramidModel( options, () => 0.5 );

		const hidden = cpuModel.decoder.layers[ 0 ];
		const output = cpuModel.decoder.layers[ 1 ];

		expect( hidden.activation ).toBe( 'hgelu' );
		expect( hidden.inputSize ).toBe( 4 );
		expect( hidden.outputSize ).toBe( 4 );

		// Zero out the hidden layer's input weights so its pre-activation is
		// exactly its bias, regardless of grid features/LOD - then pick 4
		// biases that straddle both breakpoints within this one packed group:
		// deep in the flat branch, the middle branch (twice), and deep in the
		// identity branch.
		hidden.weights.fill( 0 );
		hidden.biases = [ - 5, - 1, 0.5, 5 ];

		// Output layer (linear, 3 outputs from 4 hidden inputs): mix all 4
		// hidden neurons across the 3 outputs so every one of them (including
		// both extreme lanes) actually reaches a checked output channel.
		// output.weights is row-major [outputIndex][inputIndex].
		output.weights = [
			1, 0, 0, 1, // out0 = hidden0 + hidden3 (both extremes)
			0, 1, 0, 0, // out1 = hidden1 (middle branch, negative side)
			0, 0, 1, 0 // out2 = hidden2 (middle branch, positive side)
		];
		output.biases = [ 0, 0, 0 ];

		const pixels = await renderRawOutputs( renderer, cpuModel, gridSize );

		// Since the hidden layer's input weights are all zero, its
		// pre-activation - and therefore the whole decoder's output - is the
		// same regardless of which grid texel/LOD feeds it. `forwardMLP` with
		// an all-zero feature vector is exactly that one expected value;
		// checking every rendered texel against it is cheap and doubles as
		// coverage that the grid-sampling plumbing doesn't itself perturb a
		// position-independent result.
		const zeroFeatures = new Array( cpuModel.grids[ 0 ].channels + 1 ).fill( 0 );
		const expected = forwardMLP( cpuModel.decoder, zeroFeatures ).output;

		for ( let i = 0; i < gridSize * gridSize; i ++ ) {

			for ( let c = 0; c < 3; c ++ ) expect( pixels[ i * 4 + c ] ).toBeCloseTo( expected[ c ], 1 );

		}

		// Sanity-check the CPU reference itself actually straddles both
		// breakpoints as intended - if the hidden layer's own hand-derived
		// hardGELU outputs ever stopped straddling both breakpoints, the
		// comparison above would pass vacuously (both paths agreeing on a
		// non-straddling case proves nothing about the bug this test targets).
		const { preActivations, activations } = forwardMLP( cpuModel.decoder, zeroFeatures );
		expect( preActivations[ 0 ] ).toEqual( [ - 5, - 1, 0.5, 5 ] );
		expect( activations[ 1 ][ 0 ] ).toBeCloseTo( 0, 5 ); // hardGELU(-5) - flat branch
		expect( activations[ 1 ][ 3 ] ).toBeCloseTo( 5, 5 ); // hardGELU(5) - identity branch
		expect( activations[ 1 ][ 1 ] ).toBeCloseTo( - 1 / 3 * 0.5, 5 ); // hardGELU(-1) - middle branch
		expect( activations[ 1 ][ 2 ] ).toBeCloseTo( 0.5 / 3 * 2, 5 ); // hardGELU(0.5) - middle branch

	} );

} );

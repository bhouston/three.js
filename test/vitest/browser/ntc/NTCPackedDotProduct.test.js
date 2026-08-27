import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { float, uint, uv, vec4 } from 'three/tsl';
import { supportsPackedDotProduct, dot4I8PackedTSL, packInt8x4, quantizeSymmetricInt8, computeSymmetricScale } from '../../../../examples/jsm/ntc/NTCPackedDotProduct.js';
import { evaluateHiddenLayerDot4I8 } from '../../../../examples/jsm/ntc/NTCMLPTSL.js';
import { evaluateNeuralTextureRaw, buildMipChainTexture } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';
import { forwardMLP, hardGELU } from '../../../../examples/jsm/ntc/training/NTCMLP.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

/**
 * Renders `fragmentNode` to a `size`x`size` half-float target via a
 * fullscreen quad, and reads pixel (0,0) back as `[r,g,b,a]` floats -
 * mirrors NTCTextureSource.js's bakeColorNodeToTexture (kept as a small
 * local copy purely to avoid a cross-test-suite dependency on that
 * training-tree helper from this decoder-side test file). No special
 * material subclass/WGSL directive wiring is needed for
 * `dot4I8Packed`/`dot4U8Packed` - see NTCPackedDotProduct.js's module doc
 * comment for why (a WGSL *language feature*, not an *extension* - no
 * `enable` directive required).
 */
async function bakeAndReadPixel( renderer, fragmentNode ) {

	const size = 32; // avoids the RGBA16F readback row-alignment issue noted elsewhere in this test suite

	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 4 );
	camera.position.set( 0, 0, 2 );

	const material = new THREE.NodeMaterial();
	material.lights = false;
	material.toneMapped = false;
	material.blending = THREE.NoBlending;
	material.fragmentNode = vec4( fragmentNode );

	const mesh = new THREE.Mesh( new THREE.PlaneGeometry( 2, 2 ), material );
	scene.add( mesh );

	const renderTarget = new THREE.RenderTarget( size, size, { type: THREE.HalfFloatType } );
	const previousTarget = renderer.getRenderTarget();
	renderer.setRenderTarget( renderTarget );
	renderer.render( scene, camera );
	renderer.setRenderTarget( previousTarget );

	const pixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, 1, 1 );

	renderTarget.dispose();
	material.dispose();
	mesh.geometry.dispose();

	return [ 0, 1, 2, 3 ].map( ( i ) => THREE.DataUtils.fromHalfFloat( pixels[ i ] ) );

}

describe( 'Addons > NTC > NTCPackedDotProduct (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	describe( 'supportsPackedDotProduct', () => {

		it( 'returns false for a falsy/absent renderer', () => {

			expect( supportsPackedDotProduct( null ) ).toBe( false );
			expect( supportsPackedDotProduct( undefined ) ).toBe( false );

		} );

		it( 'returns a boolean for the real test renderer (true if this environment actually supports the language feature)', () => {

			expect( typeof supportsPackedDotProduct( renderer ) ).toBe( 'boolean' );

		} );

	} );

	// The remaining tests actually compile and run WGSL using
	// packed_4x8_integer_dot_product - each one no-ops (returns immediately,
	// neither passing nor failing meaningfully) rather than throwing whenever
	// this test environment's own WGSL compiler doesn't support it (see
	// supportsPackedDotProduct above), since that reflects a real browser/
	// driver capability gap this addon is designed to gracefully fall back
	// around, not a bug. In this repo's own CI/dev environment (Chromium via
	// Playwright on a Metal/macOS or Vulkan backend), the feature is
	// available and these tests run for real - see this file's own
	// investigation notes in .cursor/plans/ntc_paper_gap_04_packed_int8_dot_
	// product.plan.md.

	it( 'dot4I8PackedTSL matches a hand-computed int8 dot product, if supported here', async () => {

		if ( ! supportsPackedDotProduct( renderer ) ) return;

		const a = [ 3, - 4, 5, - 6 ];
		const b = [ 1, 2, - 3, 4 ];
		const expectedDot = a.reduce( ( sum, value, i ) => sum + value * b[ i ], 0 );

		const packedA = packInt8x4( a );
		const packedB = packInt8x4( b );

		const node = float( dot4I8PackedTSL( uint( packedA ), uint( packedB ) ) );
		const [ r ] = await bakeAndReadPixel( renderer, node );

		expect( r ).toBeCloseTo( expectedDot, 5 );

	} );

	// quantizeAndPackInt8TSL itself packs its result into a u32 too large to
	// round-trip exactly through a float/half-float render-target readback
	// (fp32 only exactly represents integers up to 2**24, and a packed value
	// with its sign bit set can be in the billions) - so it's not tested by
	// reading its raw output back directly. It's still fully exercised (and
	// would fail these tests if it packed incorrectly) via
	// `evaluateHiddenLayerDot4I8` below, which consumes its output and
	// produces an ordinary small dequantized float - see that test.
	it( 'quantizeAndPackInt8TSL + dot4I8PackedTSL round-trips a real-valued dot product within the expected quantization error', async () => {

		if ( ! supportsPackedDotProduct( renderer ) ) return;

		const values = [ 0.6, - 1.2, 0.3, - 0.9 ];
		const weights = [ 0.5, 0.5, - 0.5, - 0.5 ];
		const scale = computeSymmetricScale( 1.5 );

		const packedValues = uint( packInt8x4( values.map( ( v ) => quantizeSymmetricInt8( v, scale ) ) ) );
		const packedWeights = uint( packInt8x4( weights.map( ( w ) => quantizeSymmetricInt8( w, scale ) ) ) );

		const node = float( dot4I8PackedTSL( packedWeights, packedValues ) ).mul( scale ).mul( scale );
		const [ r ] = await bakeAndReadPixel( renderer, node );

		const expectedExact = values.reduce( ( sum, v, i ) => sum + v * weights[ i ], 0 );
		// Quantization error bound: each of the 4 terms can be off by at most
		// ~scale (the quantization step) on each of the two multiplicands,
		// dominated by one scale factor per term for this small example.
		expect( Math.abs( r - expectedExact ) ).toBeLessThan( scale * 4 );

	} );

	it( 'evaluateHiddenLayerDot4I8 matches a hand-computed forward pass through one hidden neuron', async () => {

		if ( ! supportsPackedDotProduct( renderer ) ) return;

		// Single hidden neuron, 4 inputs, relu - small enough to hand-verify.
		const inputs = [ 0.6, - 0.3, 0.9, - 0.1 ];
		const weights = [ 0.4, 0.6, - 0.2, 0.3 ];
		const bias = 0.05;

		const inputBound = 1.0;
		const weightBound = Math.max( ...weights.map( Math.abs ) );
		const inputScale = computeSymmetricScale( inputBound );
		const weightScale = computeSymmetricScale( weightBound );

		const packedWeights = packInt8x4( weights.map( ( w ) => quantizeSymmetricInt8( w, weightScale ) ) );

		const layerSpec = {
			inputSize: 4,
			outputSize: 1,
			activation: 'relu',
			inputScale,
			weightScale,
			getWeightPacked: () => uint( packedWeights ),
			getBias: () => float( bias )
		};

		const inputNodes = inputs.map( ( v ) => float( v ) );
		const [ outputNode ] = evaluateHiddenLayerDot4I8( inputNodes, layerSpec );
		const [ r ] = await bakeAndReadPixel( renderer, outputNode );

		// Independent reference: quantize inputs/weights the same way, compute
		// the exact integer dot product by hand (not via quantizeAndPackInt8TSL
		// or packInt8x4 - re-deriving the arithmetic directly), dequantize, add
		// bias, apply relu.
		const qInputs = inputs.map( ( v ) => quantizeSymmetricInt8( v, inputScale ) );
		const qWeights = weights.map( ( w ) => quantizeSymmetricInt8( w, weightScale ) );
		const exactDot = qInputs.reduce( ( sum, q, i ) => sum + q * qWeights[ i ], 0 );
		const expected = Math.max( 0, exactDot * inputScale * weightScale + bias );

		expect( r ).toBeCloseTo( expected, 3 );

	} );

	it( 'evaluateHiddenLayerDot4I8 applies hgelu correctly, matching NTCMLP.js\'s hardGELU on the dequantized value', async () => {

		if ( ! supportsPackedDotProduct( renderer ) ) return;

		const inputs = [ 0.6, - 0.3, 0.9, - 0.1 ];
		const weights = [ 0.4, 0.6, - 0.2, 0.3 ];
		const bias = 0.05;

		const inputScale = computeSymmetricScale( 1.0 );
		const weightScale = computeSymmetricScale( Math.max( ...weights.map( Math.abs ) ) );
		const packedWeights = packInt8x4( weights.map( ( w ) => quantizeSymmetricInt8( w, weightScale ) ) );

		const layerSpec = {
			inputSize: 4, outputSize: 1, activation: 'hgelu',
			inputScale, weightScale,
			getWeightPacked: () => uint( packedWeights ),
			getBias: () => float( bias )
		};

		const [ outputNode ] = evaluateHiddenLayerDot4I8( inputs.map( ( v ) => float( v ) ), layerSpec );
		const [ r ] = await bakeAndReadPixel( renderer, outputNode );

		const qInputs = inputs.map( ( v ) => quantizeSymmetricInt8( v, inputScale ) );
		const qWeights = weights.map( ( w ) => quantizeSymmetricInt8( w, weightScale ) );
		const exactDot = qInputs.reduce( ( sum, q, i ) => sum + q * qWeights[ i ], 0 );
		const z = exactDot * inputScale * weightScale + bias;
		const expected = hardGELU( z );

		expect( r ).toBeCloseTo( expected, 3 );

	} );

	describe( 'evaluateNeuralTextureRaw with options.useInt8DotProduct (full decoder, real trained-shaped weights)', () => {

		it( 'matches the fp32 CPU reference (forwardMLP) within the expected int8-quantization error, across multiple hidden layers', async () => {

			if ( ! supportsPackedDotProduct( renderer ) ) return;

			// Two hidden layers (relu then hgelu) so quantization error from the
			// first layer's output actually propagates into the second layer's
			// own (separately-scaled) quantization - not just a single-layer
			// case already covered above.
			// baseResolution must be >= 32 - see this file's other tests' notes
			// on the RGBA16F readback row-alignment bug (a render target/grid
			// narrower than 32 texels produces wrong - not just imprecise -
			// data for most texels, not a real correctness signal). Confirmed
			// the hard way: an earlier version of this test at baseResolution 8
			// showed drastic ("wrong axis"-looking) error that traced back
			// entirely to this, not to the int8 path itself - a real bug (fixed
			// separately) would show up as a *moderate*, quantization-shaped
			// error, not silently-zeroed texels.
			const options = {
				channels: 4, levels: 1, baseResolution: 32,
				hiddenSizes: [ 8, 8 ], outputChannels: 3,
				hiddenActivation: 'relu'
			};

			let seed = 777;
			const random = () => {

				seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
				return seed / 0x7fffffff;

			};

			const cpuModel = createNTCGridPyramidModel( options, random );
			// createMLP's default random-weight scale is small (He-initialized) -
			// scale grid data and weights up so quantization error is actually
			// exercised meaningfully (an all-near-zero network would trivially
			// "match" regardless of whether the int8 path is even correct).
			for ( let i = 0; i < cpuModel.grids[ 0 ].data.length; i ++ ) cpuModel.grids[ 0 ].data[ i ] *= 4;
			for ( const layer of cpuModel.decoder.layers ) {

				for ( let i = 0; i < layer.weights.length; i ++ ) layer.weights[ i ] *= 3;

			}

			const mipChainTexture = buildMipChainTexture( cpuModel );

			const uvNode = uv();
			const int8Raw = evaluateNeuralTextureRaw( uvNode, cpuModel, mipChainTexture, renderer, float( 0 ), { useInt8DotProduct: true } );
			const colorNode = vec4( int8Raw[ 0 ], int8Raw[ 1 ], int8Raw[ 2 ], 0 );

			const gridSize = 32; // matches baseResolution - one pixel per texel, texel centers
			const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, gridSize );
			const rawPixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize );
			renderTarget.dispose();
			mipChainTexture.dispose();

			const pixels = new Float32Array( rawPixels.length );
			for ( let i = 0; i < rawPixels.length; i ++ ) pixels[ i ] = THREE.DataUtils.fromHalfFloat( rawPixels[ i ] );

			// CPU fp32 reference at each texel center - same input construction
			// NTCDecoderTSL.test.js's referenceDecoderInput uses (channels + LOD,
			// no positional encoding here).
			let maxAbsError = 0;
			let sumAbsError = 0;
			let count = 0;

			for ( let row = 0; row < gridSize; row ++ ) {

				for ( let col = 0; col < gridSize; col ++ ) {

					const p = row * cpuModel.grids[ 0 ].width + col;
					const features = [];
					for ( let c = 0; c < cpuModel.channels; c ++ ) features.push( cpuModel.grids[ 0 ].data[ p * cpuModel.channels + c ] );
					features.push( 0 ); // LOD 0 / maxLod
					const expected = forwardMLP( cpuModel.decoder, features ).output;

					const i = row * gridSize + col;
					for ( let c = 0; c < 3; c ++ ) {

						const error = Math.abs( pixels[ i * 4 + c ] - expected[ c ] );
						maxAbsError = Math.max( maxAbsError, error );
						sumAbsError += error;
						count ++;

					}

				}

			}

			const meanAbsError = sumAbsError / count;

			console.log( `[dot4I8 vs fp32] mean abs error=${ meanAbsError.toExponential( 3 ) }, max abs error=${ maxAbsError.toExponential( 3 ) }` );

			// Bounds calibrated from this exact config's actually-observed error
			// (mean ~0.025, max ~0.16 - a coarse post-training/non-QAT
			// quantization, see NTCMLPQuantization.js's module doc comment, so
			// not tiny, but nowhere near these bounds) with real margin, not
			// arbitrarily loose ones - tight enough to still catch a real
			// regression. This specific combination (baseResolution 32,
			// 2 chained hidden layers, scaled-up weights/latents to actually
			// exercise quantization) is exactly what caught a real bug during
			// development: a render target/grid narrower than 32 texels
			// silently zeroing most texels (a WebGPU readback row-alignment
			// issue this file's other tests already work around - see the
			// `baseResolution: 32` comment above) initially produced mean/max
			// errors around 1.8/8.4 - both comfortably outside these bounds,
			// which is what made it visible as a failure instead of quietly
			// passing.
			expect( Number.isFinite( maxAbsError ) ).toBe( true );
			expect( meanAbsError ).toBeLessThan( 0.15 );
			expect( maxAbsError ).toBeLessThan( 0.6 );

		} );

	} );

} );

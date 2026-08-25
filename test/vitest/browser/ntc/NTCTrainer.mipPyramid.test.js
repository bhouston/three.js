import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { float, fract, step, uv, vec3 } from 'three/tsl';
import { NTCTrainer } from '../../../../examples/jsm/ntc/training/NTCTrainer.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { buildLevelTextures, evaluateNeuralTextureRaw } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// End-to-end coverage for mip-pyramid-aware training (see NTCMipPyramid.js /
// NTCGridPyramidModel.js / NTCGPUComputeTSL.js / NTCDecoderTSL.js), using the
// same real-WebGPU render+readback pattern as NTCDecoderTSL.test.js. Unlike
// NTCTrainer.convergence.test.js/NTCTrainer.sign-convergence.test.js (which
// deliberately pin `enableMipPyramid: false` - see their own comments - since
// they check unrelated kernel/convergence properties against a texture with
// no mip chain), this file exercises the feature itself: it trains against a
// high-frequency stripe pattern (well above the Nyquist limit of the
// pyramid's coarser grid levels) and checks that the trained decoder, when
// asked to reconstruct at a coarse LOD, actually reproduces a low-variance,
// ~0.5-mean (properly anti-aliased) result - not the aliased fine-detail
// stripe pattern a mip-0-only-trained model would produce at every LOD.
describe( 'Addons > NTC > NTCTrainer mip-pyramid training (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	/** Reads back an R-channel raster (half-float RGBA readback) as plain numbers. */
	async function readRedChannel( colorNode, resolution ) {

		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, resolution );
		const pixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, resolution, resolution );
		renderTarget.dispose();

		const values = new Array( resolution * resolution );
		for ( let i = 0; i < values.length; i ++ ) values[ i ] = THREE.DataUtils.fromHalfFloat( pixels[ i * 4 ] );

		return values;

	}

	function mean( values ) {

		return values.reduce( ( a, b ) => a + b, 0 ) / values.length;

	}

	function variance( values ) {

		const m = mean( values );
		return values.reduce( ( a, b ) => a + ( b - m ) * ( b - m ), 0 ) / values.length;

	}

	it( 'trains a decoder whose coarse-LOD reconstruction anti-aliases a high-frequency stripe pattern (low variance, ~0.5 mean) while its fine-LOD reconstruction keeps the full-contrast stripes', async () => {

		// 8 stripe cycles across a 64x64 bake - well above the Nyquist limit of
		// this model's coarser grid levels/mips, so a mip-0-only-trained model
		// would alias it at any LOD, and a correctly mip-pyramid-trained one
		// should instead reconstruct a smoothed, near-constant result once LOD
		// is high enough that the stripe frequency has been fully prefiltered
		// away in the training targets (real hardware-generated mips of the
		// baked source texture - see NTCTextureSource.js's bakeColorNodeToTexture
		// `generateMipmaps: true`).
		const bakeResolution = 64;
		const period = 8;
		const stripe = step( 0.5, fract( uv().x.mul( period ) ) );
		const stripeColorNode = vec3( stripe, stripe, stripe );

		const sourceRenderTarget = await bakeColorNodeToTexture( renderer, stripeColorNode, bakeResolution );

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 3,
			baseResolution: 16,
			growthFactor: 2,
			hiddenSizes: [ 16, 16 ],
			outputChannels: 3,
			batchSize: 2048,
			iterations: 800,
			learningRate: 0.02,
			seed: 1
			// enableMipPyramid defaults to true (NTCTrainer.DEFAULT_OPTIONS) -
			// deliberately not overridden here, this test is what exercises
			// that default.
		} );

		const result = await trainer.train( { renderer, sourceTextures: [ sourceRenderTarget.texture ] } );
		sourceRenderTarget.dispose();

		const cpuModel = result.cpuModel;

		expect( cpuModel.mipPyramid ).not.toBeNull();
		expect( cpuModel.mipPyramid.textureResolution ).toBe( bakeResolution );
		expect( cpuModel.decoder.layers[ 0 ].inputSize ).toBe( 3 * 4 + 1 );

		const levelTextures = buildLevelTextures( cpuModel );

		const fineOutputs = evaluateNeuralTextureRaw( uv(), cpuModel, levelTextures, null, float( 0 ) );
		const coarseOutputs = evaluateNeuralTextureRaw( uv(), cpuModel, levelTextures, null, float( cpuModel.mipPyramid.maxLod ) );

		const fineValues = await readRedChannel( vec3( fineOutputs[ 0 ], fineOutputs[ 1 ], fineOutputs[ 2 ] ), 32 );
		const coarseValues = await readRedChannel( vec3( coarseOutputs[ 0 ], coarseOutputs[ 1 ], coarseOutputs[ 2 ] ), 32 );

		for ( const texture of levelTextures ) texture.dispose();

		const fineVariance = variance( fineValues );
		const coarseVariance = variance( coarseValues );
		const coarseMean = mean( coarseValues );

		// The core anti-aliasing claim: reconstructing at the coarsest LOD is
		// dramatically flatter than reconstructing at LOD 0 - i.e. the decoder
		// actually learned to drop the high-frequency stripe detail as LOD
		// increases, rather than reproducing the same aliased pattern at every
		// LOD (which is what a non-mip-pyramid-aware model does today).
		expect( coarseVariance ).toBeLessThan( fineVariance * 0.35 );

		// And it should land near the pattern's true DC average (0.5, an equal
		// black/white split) - not just "lower variance than the fine
		// reconstruction" but specifically the *correct* prefiltered value.
		expect( coarseMean ).toBeGreaterThan( 0.3 );
		expect( coarseMean ).toBeLessThan( 0.7 );

		// Sanity: the fine (LOD 0) reconstruction should still show real
		// contrast - confirms the low coarse-LOD variance above reflects
		// genuine LOD-conditioned fading, not simply an undertrained/collapsed
		// decoder that outputs a flat value regardless of LOD.
		expect( fineVariance ).toBeGreaterThan( 0.05 );

	}, 60000 );

} );

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { float, fract, step, uv, vec3 } from 'three/tsl';
import { NTCTrainer } from '../../../../examples/jsm/ntc/training/NTCTrainer.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { buildMipChainTexture, evaluateNeuralTextureRaw } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// End-to-end coverage for the mip-pyramid-of-feature-grids design (see
// NTCGridPyramidModel.js / NTCMipBands.js / NTCGPUComputeTSL.js /
// NTCDecoderTSL.js), using the same real-WebGPU render+readback pattern as
// NTCDecoderTSL.test.js. Two properties matter here and are checked
// separately:
//
//  1. The decoded output genuinely varies with UV at a fixed LOD - i.e. the
//     runtime UV query is real per-pixel sampling, not something that
//     silently collapsed to a constant regardless of screen position.
//  2. Reconstructing at a coarse LOD produces a low-variance, ~0.5-mean
//     (properly anti-aliased) result for a pattern well above the pyramid's
//     representable frequency, while the fine LOD keeps full contrast - the
//     actual anti-aliasing payoff of training across the mip range.
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

	it( 'trains a decoder whose output genuinely varies with UV, and whose coarse-LOD reconstruction anti-aliases a high-frequency stripe pattern (low variance, ~0.5 mean) while its fine-LOD reconstruction keeps the full-contrast stripes', async () => {

		// 8 stripe cycles across a 64x64 bake - well above the Nyquist limit of
		// this model's coarser stored levels, so a properly mip-aware decoder
		// should reconstruct a smoothed, near-constant result at coarse LOD
		// (the training target - see NTCTextureSource.js's
		// bakeColorNodeToTexture `generateMipmaps: true`) while keeping full
		// contrast at LOD 0.
		const bakeResolution = 64;
		const period = 8;
		const stripe = step( 0.5, fract( uv().x.mul( period ) ) );
		const stripeColorNode = vec3( stripe, stripe, stripe );

		// generateMipmaps: true - this is the training source texture, which
		// training samples at LOD > 0 (see bakeColorNodeToTexture's doc
		// comment on its default).
		const sourceRenderTarget = await bakeColorNodeToTexture( renderer, stripeColorNode, bakeResolution, { generateMipmaps: true } );

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 3,
			// Deliberately left unset - NTCGridPyramidModel.js defaults it to
			// the real resolved source texture resolution (bakeResolution, via
			// NTCTrainer.js's resolveSourceTextureResolution), which is exactly
			// the scenario this test needs to exercise (see this file's
			// module doc comment / the bug this default was added to fix).
			hiddenSizes: [ 16, 16 ],
			outputChannels: 3,
			batchSize: 2048,
			iterations: 800,
			learningRate: 0.02,
			seed: 1
		} );

		const result = await trainer.train( { renderer, sourceTextures: [ sourceRenderTarget.texture ] } );
		sourceRenderTarget.dispose();

		const cpuModel = result.cpuModel;

		expect( cpuModel.maxLod ).toBe( Math.ceil( Math.log2( bakeResolution ) ) );
		expect( cpuModel.decoder.layers[ 0 ].inputSize ).toBe( cpuModel.channels + 1 );

		const mipChainTexture = buildMipChainTexture( cpuModel );

		const fineOutputs = evaluateNeuralTextureRaw( uv(), cpuModel, mipChainTexture, null, float( 0 ) );
		const coarseOutputs = evaluateNeuralTextureRaw( uv(), cpuModel, mipChainTexture, null, float( cpuModel.maxLod ) );

		const fineValues = await readRedChannel( vec3( fineOutputs[ 0 ], fineOutputs[ 1 ], fineOutputs[ 2 ] ), 32 );
		const coarseValues = await readRedChannel( vec3( coarseOutputs[ 0 ], coarseOutputs[ 1 ], coarseOutputs[ 2 ] ), 32 );

		mipChainTexture.dispose();

		const fineVariance = variance( fineValues );
		const coarseVariance = variance( coarseValues );
		const coarseMean = mean( coarseValues );

		// Property 1: the runtime UV query is real - the fine (LOD 0)
		// reconstruction shows genuine per-pixel contrast, not a value that
		// collapsed to the same constant everywhere on the quad regardless of
		// UV.
		expect( fineVariance ).toBeGreaterThan( 0.05 );

		// Property 2: the core anti-aliasing claim - reconstructing at the
		// coarsest LOD is dramatically flatter than at LOD 0, and lands near
		// the pattern's true DC average (0.5, an equal black/white split).
		expect( coarseVariance ).toBeLessThan( fineVariance * 0.35 );
		expect( coarseMean ).toBeGreaterThan( 0.3 );
		expect( coarseMean ).toBeLessThan( 0.7 );

	}, 60000 );

	it( 'reconstructs continuously across a stored-level band boundary - no discontinuous "pop" as LOD crosses it', async () => {

		// A gradient-plus-fine-detail pattern gives every stored level
		// something genuinely different to learn (unlike a flat color, which
		// would make every level converge to the same trivial output and
		// hide a real discontinuity).
		const bakeResolution = 64;
		const period = 6;
		const detail = step( 0.5, fract( uv().x.mul( period ).add( uv().y.mul( period ) ) ) );
		const patternColorNode = vec3( uv().x, uv().y, detail );

		const sourceRenderTarget = await bakeColorNodeToTexture( renderer, patternColorNode, bakeResolution, { generateMipmaps: true } );

		// levels: 3, mipsPerLevel (default 2) -> the boundary between the
		// first two stored levels' bands sits at physical mip 2 (see
		// NTCMipBands.js's `selectFeatureLevel`) - this is what
		// `evaluateNeuralTextureRaw`'s old per-level equality-mask selection
		// used to switch discontinuously across.
		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 3,
			hiddenSizes: [ 16, 16 ],
			outputChannels: 3,
			batchSize: 1024,
			iterations: 400,
			learningRate: 0.02,
			seed: 2
		} );

		const result = await trainer.train( { renderer, sourceTextures: [ sourceRenderTarget.texture ] } );
		sourceRenderTarget.dispose();

		const cpuModel = result.cpuModel;
		const boundaryLod = cpuModel.mipsPerLevel;

		const mipChainTexture = buildMipChainTexture( cpuModel );

		const sampleMeanAt = async ( lod ) => {

			const outputs = evaluateNeuralTextureRaw( uv(), cpuModel, mipChainTexture, null, float( lod ) );
			const values = await readRedChannel( vec3( outputs[ 0 ], outputs[ 1 ], outputs[ 2 ] ), 32 );
			return mean( values );

		};

		const step_ = 0.05;
		const farBefore = await sampleMeanAt( boundaryLod - 4 * step_ );
		const before = await sampleMeanAt( boundaryLod - step_ );
		const after = await sampleMeanAt( boundaryLod + step_ );

		mipChainTexture.dispose();

		// The change across a small +-step straddling the boundary should be
		// in the same ballpark as the change across an equally small step
		// entirely inside one band - not a discontinuous multiple of it, which
		// is what the old hard 0/1 level-equality switch produced (an
		// arbitrarily large jump concentrated at a single LOD value, since the
		// two neighboring levels are trained fully independently).
		const boundaryJump = Math.abs( after - before );
		const withinBandJump = Math.abs( before - farBefore );

		expect( boundaryJump ).toBeLessThan( withinBandJump * 5 + 0.05 );

	}, 30000 );

} );

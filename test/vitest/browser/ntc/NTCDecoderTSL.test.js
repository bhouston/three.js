import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { float, uv, vec4 } from 'three/tsl';
import { evaluateNeuralTextureRaw, buildMipChainTexture } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { selectFeatureLevel } from '../../../../examples/jsm/ntc/NTCMipBands.js';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';
import { forwardMLP } from '../../../../examples/jsm/ntc/training/NTCMLP.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// evaluateNeuralTextureRaw's whole job at inference time is: sample one real
// GPU mipmap-chain texture (buildMipChainTexture, see NTCHalfFloatTexture.js)
// once via hardware trilinear filtering at a given LOD, and run that tap -
// plus the normalized LOD itself - through the trained MLP decoder, all as a
// TSL node graph. The only way to know that graph computes the *same* thing
// as NTCGridPyramidModel.js's CPU decoder (the thing actually trained, and
// the thing any offline tooling would use to sanity-check a model) is to run
// both on identical inputs and compare, which is what this file does (see
// the roadmap entry in test/vitest/README.md for this exact cross-check
// pattern).
//
// To make the comparison exact rather than approximate, every test samples
// exactly at latent-grid texel centers: with LinearFilter + RepeatWrapping,
// bilinear sampling at a texel center degenerates to that texel's own value
// (weight 1), so the only source of GPU/CPU divergence is the deliberate
// half-float storage of the mip chain texture (buildMipChainTexture), which
// is accounted for by round-tripping the CPU reference input through
// THREE.DataUtils.toHalfFloat/fromHalfFloat before feeding it to forwardMLP.
// A fractional LOD additionally blends between the two bracketing physical
// mips' texel values *before* forwardMLP (the network is nonlinear, so
// blending post-decode would not be equivalent) - see the trilinear-blend
// test below.

function makeRandom( seedMultiplier ) {

	let calls = 0;
	return () => {

		calls += 1;
		return ( Math.sin( calls * seedMultiplier ) + 1 ) / 2;

	};

}

function readHalfFloatPixels( pixels ) {

	const out = new Float32Array( pixels.length );
	for ( let i = 0; i < pixels.length; i ++ ) out[ i ] = THREE.DataUtils.fromHalfFloat( pixels[ i ] );
	return out;

}

// Half-float-rounds a latent grid value the same way buildMipChainTexture's
// Uint16Array packing does, so the CPU reference input matches exactly what
// the GPU decoder actually reads back from the level texture.
function toHalfPrecision( value ) {

	return THREE.DataUtils.fromHalfFloat( THREE.DataUtils.toHalfFloat( value ) );

}

// Overwrites every texel of a CPU-model grid with the same per-channel
// values - see the trilinear-blend test below for why.
function fillGridConstant( grid, values ) {

	for ( let p = 0; p < grid.width * grid.height; p ++ ) {

		for ( let c = 0; c < grid.channels; c ++ ) grid.data[ p * grid.channels + c ] = values[ c ];

	}

}

// Builds the plain-JS reference input vector for the decoder at texel
// (col, row) of the grid level `lod` selects: the selected level's
// (half-float-rounded) channels, followed by the normalized LOD - in the
// same order evaluateNeuralTextureRaw builds them.
function referenceDecoderInput( cpuModel, lod, col, row ) {

	const levelIndex = selectFeatureLevel( lod, cpuModel.grids.length, cpuModel.mipsPerLevel );
	const grid = cpuModel.grids[ levelIndex ];
	const p = row * grid.width + col;

	const features = [];
	for ( let c = 0; c < grid.channels; c ++ ) features.push( toHalfPrecision( grid.data[ p * grid.channels + c ] ) );
	features.push( lod / cpuModel.maxLod );

	return features;

}

async function renderRawOutputs( renderer, cpuModel, lod, gridSize ) {

	const mipChainTexture = buildMipChainTexture( cpuModel );
	const raw = evaluateNeuralTextureRaw( uv(), cpuModel, mipChainTexture, null, float( lod ) );
	// TSL's vec4() only spreads its own varargs into components - handing it
	// the raw JS array as a single argument would treat that array as one
	// opaque const value instead of 4 components, so every component beyond
	// the ConvertNode's coercion reads back as 0. Spread explicitly, padding
	// to 4 with a constant 0 for outputChannels < 4.
	const components = [ raw[ 0 ], raw[ 1 ], raw[ 2 ], raw[ 3 ] ];
	const colorNode = vec4( components[ 0 ], components[ 1 ], components[ 2 ] ?? 0, components[ 3 ] ?? 0 );

	const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, gridSize );
	const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );

	renderTarget.dispose();
	mipChainTexture.dispose();

	return pixels;

}

function expectMatchesCpuForward( pixels, cpuModel, lod, gridSize, componentCount = 3 ) {

	for ( let row = 0; row < gridSize; row ++ ) {

		for ( let col = 0; col < gridSize; col ++ ) {

			const input = referenceDecoderInput( cpuModel, lod, col, row );
			const expected = forwardMLP( cpuModel.decoder, input ).output;

			const i = row * gridSize + col;
			for ( let c = 0; c < componentCount; c ++ ) expect( pixels[ i * 4 + c ] ).toBeCloseTo( expected[ c ], 2 );

		}

	}

}

describe( 'Addons > NTC > NTCDecoderTSL (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	it( 'matches NTCGridPyramidModel.js\'s CPU forward pass at every latent-grid texel (single level, LOD 0)', async () => {

		const gridSize = 32;
		const options = { channels: 4, levels: 1, baseResolution: gridSize, hiddenSizes: [ 4 ], outputChannels: 3 };
		const cpuModel = createNTCGridPyramidModel( options, makeRandom( 1.7 ) );

		expect( cpuModel.decoder.layers[ 0 ].inputSize ).toBe( options.channels + 1 );

		const pixels = await renderRawOutputs( renderer, cpuModel, 0, gridSize );

		expectMatchesCpuForward( pixels, cpuModel, 0, gridSize );

	} );

	it( 'selects the correct stored grid level per LOD, matching NTCMipBands.selectFeatureLevel', async () => {

		// mipsPerLevel: 1 gives a plain per-mip halving chain (64, 32) - LOD 0
		// selects level 0 (64x64), LOD 1 selects level 1 (32x32). Each is
		// checked at its own resolution, texel-exact. Both resolutions are
		// deliberately kept at >= 32: a render target narrower than 32 texels
		// has an RGBA16F row byte size below 256 (16 texels * 4 channels * 2
		// bytes = 128), which has been observed to hit a WebGPU pixel-readback
		// row-alignment bug unrelated to this addon (reproduces with a bare
		// `uv()` bake, no texture sampling involved at all) - not something to
		// work around here, just avoided by testing at safely-aligned sizes.
		const finestSize = 64;
		const options = { channels: 2, levels: 2, baseResolution: finestSize, mipsPerLevel: 1, hiddenSizes: [ 4 ], outputChannels: 3 };
		const cpuModel = createNTCGridPyramidModel( options, makeRandom( 0.9 ) );

		expect( cpuModel.resolutions ).toEqual( [ 64, 32 ] );
		expect( selectFeatureLevel( 0, cpuModel.grids.length, cpuModel.mipsPerLevel ) ).toBe( 0 );
		expect( selectFeatureLevel( 1, cpuModel.grids.length, cpuModel.mipsPerLevel ) ).toBe( 1 );

		const finePixels = await renderRawOutputs( renderer, cpuModel, 0, finestSize );
		expectMatchesCpuForward( finePixels, cpuModel, 0, finestSize );

		const coarseSize = 32;
		const coarsePixels = await renderRawOutputs( renderer, cpuModel, 1, coarseSize );
		expectMatchesCpuForward( coarsePixels, cpuModel, 1, coarseSize );

	} );

	it( 'evaluateNeuralTextureRaw\'s 4th output channel matches the CPU forward pass (not just the first 3)', async () => {

		const gridSize = 32;
		const options = { channels: 3, levels: 1, baseResolution: gridSize, hiddenSizes: [ 4 ], outputChannels: 4 };
		const cpuModel = createNTCGridPyramidModel( options, makeRandom( 3.1 ) );

		const pixels = await renderRawOutputs( renderer, cpuModel, 0, gridSize );

		expectMatchesCpuForward( pixels, cpuModel, 0, gridSize, 4 );

	} );

	it( 'blends smoothly between two bracketing physical mips at a fractional LOD, including across a stored-level band boundary, matching a JS-blended feature vector fed through the (nonlinear) decoder', async () => {

		// mipsPerLevel: 2 -> computeGridLevels(16, 2, 2) stores [16, 4]; the
		// physical mip chain built from that is [16, 8, 4, 2, 1] (see
		// buildMipChainLevels), with physical mip 0-1 covered by level 0's
		// band and mip 2-4 by level 1's (open-ended tail) band. LOD 1.5
		// therefore brackets physical mip 1 (still level 0's band - a
		// box-filtered-down copy of level 0's own data) and physical mip 2
		// (level 1's own native data) - i.e. exactly the boundary *between*
		// the two independently-trained levels, the discontinuity hardware
		// trilinear sampling is meant to smooth over (see
		// buildMipChainTexture's doc comment).
		const options = { channels: 2, levels: 2, baseResolution: 16, mipsPerLevel: 2, hiddenSizes: [ 4 ], outputChannels: 3 };
		const cpuModel = createNTCGridPyramidModel( options, makeRandom( 2.3 ) );

		// Both stored levels are overwritten with a flat per-channel constant:
		// a box-filtered average of a constant field is that same constant
		// again, so *every* physical mip in a level's band evaluates to one
		// exactly-known value, independent of UV or which texels a bilinear
		// tap lands on. That isolates exactly what this test checks - the
		// linear blend *between* the two bracketing mips that a fractional
		// LOD drives - from ordinary bilinear/UV concerns already covered by
		// the other tests in this file.
		const levelAValue = [ 0.2, - 0.35 ];
		const levelBValue = [ 0.6, 0.1 ];
		fillGridConstant( cpuModel.grids[ 0 ], levelAValue );
		fillGridConstant( cpuModel.grids[ 1 ], levelBValue );

		const lod = 1.5;
		const gridSize = 32;

		const mipChainTexture = buildMipChainTexture( cpuModel );
		const raw = evaluateNeuralTextureRaw( uv(), cpuModel, mipChainTexture, null, float( lod ) );
		const colorNode = vec4( raw[ 0 ], raw[ 1 ], raw[ 2 ], 0 );

		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, gridSize );
		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );
		renderTarget.dispose();
		mipChainTexture.dispose();

		// The CPU reference blends the two bracketing mips' (half-float-
		// rounded, matching what the GPU actually reads back from the
		// texture) constant values *before* the decoder - not the decoder's
		// two separately-evaluated outputs after - since the decoder is a
		// nonlinear MLP and those are not equivalent.
		const blendWeight = lod - Math.floor( lod );
		const blendedFeatures = levelAValue.map( ( a, c ) => {

			const halfA = toHalfPrecision( a );
			const halfB = toHalfPrecision( levelBValue[ c ] );
			return halfA + ( halfB - halfA ) * blendWeight;

		} );
		blendedFeatures.push( lod / cpuModel.maxLod );

		const expected = forwardMLP( cpuModel.decoder, blendedFeatures ).output;

		for ( let i = 0; i < gridSize * gridSize; i ++ ) {

			for ( let c = 0; c < 3; c ++ ) expect( pixels[ i * 4 + c ] ).toBeCloseTo( expected[ c ], 2 );

		}

	} );

} );

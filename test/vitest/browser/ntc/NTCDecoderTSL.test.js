import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { uv, vec4 } from 'three/tsl';
import { evaluateNeuralTextureRaw, buildLevelTextures } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/NTCGridPyramidModel.js';
import { forwardMLP } from '../../../../examples/jsm/ntc/NTCMLP.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/NTCTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// evaluateNeuralTextureRaw's whole job at inference time is: sample the
// trained multiresolution latent grid (as ordinary hardware-filtered
// textures) at a UV, concatenate the taps, and run that through the trained
// MLP decoder - all as a TSL node graph. The only way to know that graph
// computes the *same* thing as NTCGridPyramidModel.js's CPU decoder (the
// thing actually trained, and the thing any offline tooling would use to
// sanity-check a model) is to run both on identical inputs and compare,
// which is what this file does (see the roadmap entry in
// test/vitest/README.md for this exact cross-check pattern).
//
// To make the comparison exact rather than approximate, every test samples
// exactly at latent-grid texel centers: with LinearFilter + RepeatWrapping,
// bilinear sampling at a texel center degenerates to that texel's own value
// (weight 1), so the only source of GPU/CPU divergence is the deliberate
// half-float storage of the latent grid (buildLevelTextures), which is
// accounted for by round-tripping the CPU reference input through
// THREE.DataUtils.toHalfFloat/fromHalfFloat before feeding it to forwardMLP.

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

// Half-float-rounds a latent grid value the same way buildLevelTextures'
// Uint16Array packing does, so the CPU reference input matches exactly what
// the GPU decoder actually reads back from the level texture.
function toHalfPrecision( value ) {

	return THREE.DataUtils.fromHalfFloat( THREE.DataUtils.toHalfFloat( value ) );

}

// Builds the plain-JS reference input vector for the decoder at grid texel
// (col, row): the concatenated (half-float-rounded) latent features across
// every level, in the same order evaluateNeuralTextureRaw concatenates them.
function referenceDecoderInput( cpuModel, gridSize, col, row ) {

	const features = [];

	for ( const grid of cpuModel.grids ) {

		const p = row * grid.width + col;
		for ( let c = 0; c < grid.channels; c ++ ) {

			features.push( toHalfPrecision( grid.data[ p * grid.channels + c ] ) );

		}

	}

	return features;

}

async function renderRawOutputs( renderer, cpuModel, gridSize ) {

	const levelTextures = buildLevelTextures( cpuModel );
	const raw = evaluateNeuralTextureRaw( uv(), cpuModel, levelTextures );
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
	for ( const levelTexture of levelTextures ) levelTexture.dispose();

	return pixels;

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

	it( 'matches NTCGridPyramidModel.js\'s CPU forward pass at every latent-grid texel (single level)', async () => {

		const gridSize = 32;
		const options = { channels: 4, levels: 1, baseResolution: gridSize, hiddenSizes: [ 4 ], outputChannels: 3 };
		const cpuModel = createNTCGridPyramidModel( options, makeRandom( 1.7 ) );

		const pixels = await renderRawOutputs( renderer, cpuModel, gridSize );

		for ( let row = 0; row < gridSize; row ++ ) {

			for ( let col = 0; col < gridSize; col ++ ) {

				const input = referenceDecoderInput( cpuModel, gridSize, col, row );
				const expected = forwardMLP( cpuModel.decoder, input ).output;

				const i = row * gridSize + col;
				expect( pixels[ i * 4 + 0 ] ).toBeCloseTo( expected[ 0 ], 2 );
				expect( pixels[ i * 4 + 1 ] ).toBeCloseTo( expected[ 1 ], 2 );
				expect( pixels[ i * 4 + 2 ] ).toBeCloseTo( expected[ 2 ], 2 );

			}

		}

	} );

	it( 'concatenates multiple grid levels in the same order as the CPU model (levels * channels input)', async () => {

		const gridSize = 32;
		// growthFactor: 1 forces every level to the same resolution, which
		// keeps texel-center addressing identical for both levels while still
		// exercising real multi-level concatenation order.
		const options = { channels: 2, levels: 2, baseResolution: gridSize, growthFactor: 1, hiddenSizes: [ 4 ], outputChannels: 3 };
		const cpuModel = createNTCGridPyramidModel( options, makeRandom( 0.9 ) );

		expect( cpuModel.grids.length ).toBe( 2 );
		expect( cpuModel.decoder.layers[ 0 ].inputSize ).toBe( options.levels * options.channels );

		const pixels = await renderRawOutputs( renderer, cpuModel, gridSize );

		for ( let row = 0; row < gridSize; row ++ ) {

			for ( let col = 0; col < gridSize; col ++ ) {

				const input = referenceDecoderInput( cpuModel, gridSize, col, row );
				expect( input.length ).toBe( options.levels * options.channels );

				const expected = forwardMLP( cpuModel.decoder, input ).output;

				const i = row * gridSize + col;
				expect( pixels[ i * 4 + 0 ] ).toBeCloseTo( expected[ 0 ], 2 );
				expect( pixels[ i * 4 + 1 ] ).toBeCloseTo( expected[ 1 ], 2 );
				expect( pixels[ i * 4 + 2 ] ).toBeCloseTo( expected[ 2 ], 2 );

			}

		}

	} );

	it( 'evaluateNeuralTextureRaw\'s 4th output channel matches the CPU forward pass (not just the first 3)', async () => {

		const gridSize = 32;
		const options = { channels: 3, levels: 1, baseResolution: gridSize, hiddenSizes: [ 4 ], outputChannels: 4 };
		const cpuModel = createNTCGridPyramidModel( options, makeRandom( 3.1 ) );

		const pixels = await renderRawOutputs( renderer, cpuModel, gridSize );

		for ( let row = 0; row < gridSize; row ++ ) {

			for ( let col = 0; col < gridSize; col ++ ) {

				const input = referenceDecoderInput( cpuModel, gridSize, col, row );
				const expected = forwardMLP( cpuModel.decoder, input ).output;

				const i = row * gridSize + col;
				for ( let c = 0; c < 4; c ++ ) expect( pixels[ i * 4 + c ] ).toBeCloseTo( expected[ c ], 2 );

			}

		}

	} );

} );

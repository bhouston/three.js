import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { uv, vec4 } from 'three/tsl';
import {
	NeuralTextureNodeMaterial,
	evaluateNeuralTextureRaw,
	buildLevelTextures
} from '../../../../examples/jsm/neural-texture/NeuralTextureNodeMaterial.js';
import { createNeuralTextureModel } from '../../../../examples/jsm/neural-texture/NeuralTextureModel.js';
import { triangleWaveEncode } from '../../../../examples/jsm/neural/NeuralGridModel.js';
import { forwardMLP } from '../../../../examples/jsm/neural/NeuralMLP.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/neural-texture/NeuralTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// NeuralTextureNodeMaterial's whole job at inference time is: sample the
// trained multiresolution latent grid (as ordinary hardware-filtered
// textures) at a UV, concatenate the taps (+ optionally the raw (u, v) skip
// connection), and run that through the trained MLP decoder - all as a TSL
// node graph baked into the material's fragment shader. The only way to know
// that graph computes the *same* thing as NeuralTextureModel.js's CPU
// decoder (the thing actually trained, and the thing any offline tooling
// would use to sanity-check a model) is to run both on identical inputs and
// compare, which is what this file does (see the roadmap entry in
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
// every level, in the same order evaluateNeuralTextureRaw concatenates them,
// plus the tiled positional encoding (see NeuralGridModel.triangleWaveEncode)
// when the model was trained with peOctaves > 0.
function referenceDecoderInput( cpuModel, gridSize, col, row ) {

	const features = [];

	for ( const grid of cpuModel.grids ) {

		const p = row * grid.width + col;
		for ( let c = 0; c < grid.channels; c ++ ) {

			features.push( toHalfPrecision( grid.data[ p * grid.channels + c ] ) );

		}

	}

	if ( cpuModel.peOctaves > 0 ) {

		features.push( ...triangleWaveEncode( ( col + 0.5 ) / gridSize, ( row + 0.5 ) / gridSize, cpuModel.peOctaves ) );

	}

	return features;

}

describe( 'Addons > Neural > Neural-Texture > NeuralTextureNodeMaterial (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	it( 'matches NeuralTextureModel.js\'s CPU forward pass at every latent-grid texel (single level, no positional encoding)', async () => {

		const gridSize = 32;
		const options = { channels: 4, levels: 1, baseResolution: gridSize, hiddenSizes: [ 4 ], outputChannels: 3, peOctaves: 0 };
		const cpuModel = createNeuralTextureModel( options, makeRandom( 1.7 ) );

		const material = new NeuralTextureNodeMaterial( cpuModel );

		const renderTarget = await bakeColorNodeToTexture( renderer, material.colorNode, gridSize );
		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );

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

		renderTarget.dispose();
		material.dispose();

	} );

	it( 'matches the CPU forward pass when the model was trained with tiled positional encoding', async () => {

		const gridSize = 32;
		const options = { channels: 2, levels: 1, baseResolution: gridSize, hiddenSizes: [ 3 ], outputChannels: 3, peOctaves: 2 };
		const cpuModel = createNeuralTextureModel( options, makeRandom( 2.3 ) );

		expect( cpuModel.decoder.layers[ 0 ].inputSize ).toBe( options.channels + 2 * options.peOctaves );

		const material = new NeuralTextureNodeMaterial( cpuModel );

		const renderTarget = await bakeColorNodeToTexture( renderer, material.colorNode, gridSize );
		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );

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

		renderTarget.dispose();
		material.dispose();

	} );

	it( 'concatenates multiple grid levels in the same order as the CPU model (levels * channels input)', async () => {

		const gridSize = 32;
		// growthFactor: 1 forces every level to the same resolution, which
		// keeps texel-center addressing identical for both levels while still
		// exercising real multi-level concatenation order.
		const options = { channels: 2, levels: 2, baseResolution: gridSize, growthFactor: 1, hiddenSizes: [ 4 ], outputChannels: 3, peOctaves: 0 };
		const cpuModel = createNeuralTextureModel( options, makeRandom( 0.9 ) );

		expect( cpuModel.grids.length ).toBe( 2 );
		expect( cpuModel.decoder.layers[ 0 ].inputSize ).toBe( options.levels * options.channels );

		const material = new NeuralTextureNodeMaterial( cpuModel );

		const renderTarget = await bakeColorNodeToTexture( renderer, material.colorNode, gridSize );
		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );

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

		renderTarget.dispose();
		material.dispose();

	} );

	it( 'evaluateNeuralTextureRaw\'s 4th output channel matches the CPU forward pass (not just the first 3, as the vec3 wrapper would give)', async () => {

		const gridSize = 32;
		const options = { channels: 3, levels: 1, baseResolution: gridSize, hiddenSizes: [ 4 ], outputChannels: 4 };
		const cpuModel = createNeuralTextureModel( options, makeRandom( 3.1 ) );

		const levelTextures = buildLevelTextures( cpuModel );
		const raw = evaluateNeuralTextureRaw( uv(), cpuModel, levelTextures );
		// TSL's vec4() only spreads its own varargs into components - handing
		// it the raw JS array as a single argument would treat that array as
		// one opaque const value instead of 4 components, so every component
		// beyond the ConvertNode's coercion reads back as 0. Spread explicitly.
		const colorNode = vec4( raw[ 0 ], raw[ 1 ], raw[ 2 ], raw[ 3 ] );

		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, gridSize );
		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );

		for ( let row = 0; row < gridSize; row ++ ) {

			for ( let col = 0; col < gridSize; col ++ ) {

				const input = referenceDecoderInput( cpuModel, gridSize, col, row );
				const expected = forwardMLP( cpuModel.decoder, input ).output;

				const i = row * gridSize + col;
				for ( let c = 0; c < 4; c ++ ) expect( pixels[ i * 4 + c ] ).toBeCloseTo( expected[ c ], 2 );

			}

		}

		renderTarget.dispose();
		for ( const levelTexture of levelTextures ) levelTexture.dispose();

	} );

	it( 'the diff mode output matches abs(predicted - source) * errorScale against the CPU reference', async () => {

		const gridSize = 32;
		const options = { channels: 4, levels: 1, baseResolution: gridSize, growthFactor: 2, hiddenSizes: [ 4 ], outputChannels: 3 };
		const cpuModel = createNeuralTextureModel( options, makeRandom( 4.4 ) );

		// A known, deterministic "source" texture to diff against - solid
		// values, packed the same way buildLevelTextures packs the latent
		// grid (half-float RGBA, repeat wrap, linear filter) so sampling at a
		// texel center is again exact.
		const sourceValue = [ 0.2, - 0.3, 0.4 ];
		const sourceData = new Uint16Array( gridSize * gridSize * 4 );
		for ( let p = 0; p < gridSize * gridSize; p ++ ) {

			for ( let c = 0; c < 3; c ++ ) sourceData[ p * 4 + c ] = THREE.DataUtils.toHalfFloat( sourceValue[ c ] );
			sourceData[ p * 4 + 3 ] = THREE.DataUtils.toHalfFloat( 1 );

		}

		const sourceTexture = new THREE.DataTexture( sourceData, gridSize, gridSize, THREE.RGBAFormat, THREE.HalfFloatType );
		sourceTexture.wrapS = THREE.RepeatWrapping;
		sourceTexture.wrapT = THREE.RepeatWrapping;
		sourceTexture.magFilter = THREE.LinearFilter;
		sourceTexture.minFilter = THREE.LinearFilter;
		sourceTexture.generateMipmaps = false;
		sourceTexture.needsUpdate = true;

		const errorScale = 2.5;
		const material = new NeuralTextureNodeMaterial( cpuModel, { mode: 'diff', sourceTexture, errorScale } );

		const renderTarget = await bakeColorNodeToTexture( renderer, material.colorNode, gridSize );
		const pixels = readHalfFloatPixels( await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, gridSize, gridSize ) );

		for ( let row = 0; row < gridSize; row ++ ) {

			for ( let col = 0; col < gridSize; col ++ ) {

				const input = referenceDecoderInput( cpuModel, gridSize, col, row );
				const predicted = forwardMLP( cpuModel.decoder, input ).output;

				const i = row * gridSize + col;
				for ( let c = 0; c < 3; c ++ ) {

					const expected = Math.abs( predicted[ c ] - sourceValue[ c ] ) * errorScale;
					expect( pixels[ i * 4 + c ] ).toBeCloseTo( expected, 2 );

				}

			}

		}

		renderTarget.dispose();
		material.dispose();
		sourceTexture.dispose();

	} );

	it( 'dispose() disposes exactly one texture per grid level', () => {

		const options = { channels: 4, levels: 3, baseResolution: 4, growthFactor: 2, hiddenSizes: [ 4 ], outputChannels: 3 };
		const cpuModel = createNeuralTextureModel( options, makeRandom( 5.5 ) );

		const material = new NeuralTextureNodeMaterial( cpuModel );
		expect( material.levelTextures.length ).toBe( 3 );

		const disposeSpies = material.levelTextures.map( ( levelTexture ) => vi.spyOn( levelTexture, 'dispose' ) );

		material.dispose();

		for ( const spy of disposeSpies ) expect( spy ).toHaveBeenCalledTimes( 1 );

	} );

} );

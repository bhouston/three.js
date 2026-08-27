import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { float, vec2, vec4 } from 'three/tsl';
import { NTCTrainer } from '../../../../examples/jsm/ntc/training/NTCTrainer.js';
import { evaluateNeuralTextureRaw, buildMipChainTexture } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { encodeNTC } from '../../../../examples/jsm/ntc/training/NTCManifest.js';
import { NTCLoader } from '../../../../examples/jsm/loaders/NTCLoader.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// End-to-end coverage for `positionalEncoding` (see NTCPositionalEncoding.js)
// through the full real training loop - NTCGPUComputeTSL.test.js/
// NTCDecoderTSL.test.js already cross-check the underlying math precisely
// against a CPU reference in isolation (a controlled single sample, and an
// explicit fixed UV, respectively); this file instead exercises the full
// stochastic training loop (real random UV/LOD sampling) end to end, the way
// NTCTrainer.convergence.test.js does for the rest of the trainer, and the
// full export -> load -> decode round trip through NTCManifest.js/
// NTCLoader.js - the two things a per-sample math cross-check alone can't
// catch (an actual training-loop bug only visible over many stochastic
// samples, or a manifest/loader field that isn't wired through end to end).
describe( 'Addons > NTC > NTCTrainer with positional encoding (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	function buildGradientTexture( resolution ) {

		const data = new Uint16Array( resolution * resolution * 4 );

		for ( let y = 0; y < resolution; y ++ ) {

			for ( let x = 0; x < resolution; x ++ ) {

				const u = ( x + 0.5 ) / resolution;
				const v = ( y + 0.5 ) / resolution;
				const p = ( y * resolution + x ) * 4;

				data[ p + 0 ] = THREE.DataUtils.toHalfFloat( u );
				data[ p + 1 ] = THREE.DataUtils.toHalfFloat( v );
				data[ p + 2 ] = THREE.DataUtils.toHalfFloat( 0.5 );
				data[ p + 3 ] = THREE.DataUtils.toHalfFloat( 1 );

			}

		}

		const texture = new THREE.DataTexture( data, resolution, resolution, THREE.RGBAFormat, THREE.HalfFloatType );
		texture.wrapS = THREE.RepeatWrapping;
		texture.wrapT = THREE.RepeatWrapping;
		texture.magFilter = THREE.LinearFilter;
		texture.minFilter = THREE.LinearFilter;
		texture.minFilter = THREE.LinearMipmapLinearFilter;
		texture.generateMipmaps = true;
		texture.needsUpdate = true;

		return texture;

	}

	it( 'trains without NaN/divergence and reduces loss, matching a plain (no positional encoding) run\'s stability', async () => {

		const texture = buildGradientTexture( 32 );

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 2,
			baseResolution: 8,
			hiddenSizes: [ 16, 16 ],
			outputChannels: 3,
			batchSize: 2048,
			iterations: 150,
			learningRate: 0.02,
			seed: 1,
			textureResolution: 1, // see NTCTrainer.convergence.test.js's identical setup for why
			positionalEncoding: true,
			positionalEncodingOctaves: 2
		} );

		const lossHistory = [];
		const result = await trainer.train( {
			renderer,
			sourceTextures: [ texture ],
			onProgress: ( progress ) => lossHistory.push( progress.loss )
		} );

		texture.dispose();

		expect( Number.isFinite( result.loss ) ).toBe( true );
		expect( result.cpuModel.positionalEncoding ).toBe( true );
		expect( result.cpuModel.positionalEncodingOctaves ).toBe( 2 );

		const first = lossHistory[ 0 ];
		const last = lossHistory[ lossHistory.length - 1 ];
		expect( last ).toBeLessThan( first );

		// No runaway divergence at any logged point (same invariant
		// NTCTrainer.convergence.test.js checks for the non-positional-
		// encoding path).
		for ( const loss of lossHistory ) expect( loss ).toBeLessThan( first * 5 + 1e-6 );

	}, 60000 );

	it( 'a trained positional-encoding model exports, reloads, and decodes to finite output with the widened input layer intact', async () => {

		const texture = buildGradientTexture( 16 );

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 1,
			baseResolution: 8,
			hiddenSizes: [ 8 ],
			outputChannels: 3,
			batchSize: 1024,
			iterations: 50,
			learningRate: 0.02,
			seed: 2,
			// Deliberately NOT textureResolution: 1 here (unlike the sibling
			// test above) - that forces maxLod to exactly 0 (see
			// NTCGridPyramidModel.js), which NTCLoader.js's validateManifest
			// currently rejects (`assertInteger(..., 'latents.maxLod', 1)`,
			// requiring >= 1) - a pre-existing constraint unrelated to
			// positional encoding, not something this test is meant to cover.
			textureResolution: 8,
			positionalEncoding: true,
			positionalEncodingOctaves: 2
		} );

		const result = await trainer.train( { renderer, sourceTextures: [ texture ] } );
		texture.dispose();

		const classification = { activeChannels: [], constantValues: {}, renderFlags: null };
		const manifest = encodeNTC( result.cpuModel, classification, { name: 'positional encoding round trip' } );

		expect( manifest.positionalEncoding ).toEqual( { octaves: 2 } );

		const json = JSON.parse( JSON.stringify( manifest ) );
		const loaded = new NTCLoader().parse( json );

		expect( loaded.cpuModel.positionalEncoding ).toBe( true );
		expect( loaded.cpuModel.positionalEncodingOctaves ).toBe( 2 );
		expect( loaded.cpuModel.decoder.layers[ 0 ].inputSize ).toBe( result.cpuModel.decoder.layers[ 0 ].inputSize );

		// Decode a real pixel through the loaded model's own decoder and
		// confirm it doesn't throw and produces finite output - the actual
		// consumer-facing path (NTCNodeMaterial.js) this all needs to work
		// through.
		const mipChainTexture = buildMipChainTexture( loaded.cpuModel );
		const raw = evaluateNeuralTextureRaw( vec2( 0.3, 0.6 ), loaded.cpuModel, mipChainTexture, null, float( 0 ) );

		const colorNode = vec4( raw[ 0 ], raw[ 1 ], raw[ 2 ], 0 );
		const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, 32 );
		const pixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, 32, 32 );

		renderTarget.dispose();
		mipChainTexture.dispose();

		for ( let i = 0; i < 4; i ++ ) expect( Number.isFinite( THREE.DataUtils.fromHalfFloat( pixels[ i ] ) ) ).toBe( true );

	}, 60000 );

} );

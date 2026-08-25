import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { vec2, vec4 } from 'three/tsl';
import { NTCTrainer } from '../../../../examples/jsm/ntc/training/NTCTrainer.js';
import { evaluateNeuralTextureRaw, buildLevelTextures } from '../../../../examples/jsm/ntc/NTCDecoderTSL.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { applyChannelActivation } from '../../../../examples/jsm/ntc/NTCOutputActivations.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// End-to-end regression test for the leading hypothesis behind the reported
// neural-material "brick" bug: with debug-view coloring fixed (see
// NTCFormat.test.js) and reconstructFinalNormal itself shown to
// treat +/- (dx, dy) symmetrically (see NTCNodeMaterial.test.js),
// the remaining live suspicion is that the *trained* tanh-activated `normal`
// channel doesn't actually learn negative-going offsets as reliably as
// positive-going ones - i.e. the network itself, not any surrounding
// plumbing, is the source of "some brick edges are missing".
//
// This builds a synthetic training target that deliberately reproduces the
// exact channel *layout* the real "brick" preset hits (see
// NTCSource.test.js's "packed alone" case): 3 filler sigmoid
// channels (albedo-like) + 2 tanh channels (dx, dy), 5 components total, so
// dx lands as the 4th component of the first packed render target and dy
// lands *alone* as the 1st component of a second, otherwise-zero-padded
// render target - exactly like the real material. dx alternates sign with a
// clean square-wave stripe pattern in u, dy alternates sign with the same
// pattern in v, so both positive and negative excursions are equally
// represented in the ground truth for both channels.
//
// If the network is systematically better at learning one sign than the
// other, this test fails; if it isn't, it passes and this specific
// hypothesis is ruled out by actual execution, not just code reading.
describe( 'Addons > Neural > NeuralTexture > NTCTrainer sign convergence (real WebGPU, end-to-end training)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	function stripeSign( t, period ) {

		return Math.floor( t * period ) % 2 === 0 ? 1 : - 1;

	}

	function buildStripeTexture( resolution, valueFn ) {

		const data = new Uint16Array( resolution * resolution * 4 );

		for ( let y = 0; y < resolution; y ++ ) {

			for ( let x = 0; x < resolution; x ++ ) {

				const u = ( x + 0.5 ) / resolution;
				const v = ( y + 0.5 ) / resolution;
				const [ r, g, b, a ] = valueFn( u, v );
				const p = ( y * resolution + x ) * 4;

				data[ p + 0 ] = THREE.DataUtils.toHalfFloat( r );
				data[ p + 1 ] = THREE.DataUtils.toHalfFloat( g );
				data[ p + 2 ] = THREE.DataUtils.toHalfFloat( b );
				data[ p + 3 ] = THREE.DataUtils.toHalfFloat( a );

			}

		}

		const texture = new THREE.DataTexture( data, resolution, resolution, THREE.RGBAFormat, THREE.HalfFloatType );
		texture.wrapS = THREE.RepeatWrapping;
		texture.wrapT = THREE.RepeatWrapping;
		texture.magFilter = THREE.LinearFilter;
		texture.minFilter = THREE.LinearFilter;
		texture.generateMipmaps = false;
		texture.needsUpdate = true;

		return texture;

	}

	it( 'trains a tanh channel that is alone in its own packed render target to the correct sign on both sides of a stripe pattern', async () => {

		const AMPLITUDE = 0.7;
		const PERIOD = 4;
		const resolution = 32;

		// pack0 = (fillerR, fillerG, fillerB, dx) - dx alternates sign with u.
		const pack0 = buildStripeTexture( resolution, ( u ) => [ 0.5, 0.5, 0.5, stripeSign( u, PERIOD ) * AMPLITUDE ] );
		// pack1 = (dy, 0, 0, 0) - dy alone, alternates sign with v. Mirrors
		// exactly how the real "brick" preset's normal channel lands (see
		// NTCSource.test.js).
		const pack1 = buildStripeTexture( resolution, ( u, v ) => [ stripeSign( v, PERIOD ) * AMPLITUDE, 0, 0, 0 ] );

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 3,
			baseResolution: 8,
			growthFactor: 2,
			hiddenSizes: [ 16, 16 ],
			outputChannels: 5,
			channelActivations: [ 'sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh' ],
			batchSize: 2048,
			iterations: 500,
			learningRate: 0.02,
			seed: 1,
			// This file tests channel-sign correctness against mip-0-only
			// textures (`generateMipmaps: false` above) - orthogonal to
			// mip-pyramid training (default true, see NTCTrainer.js), which
			// would sample other mip levels these textures deliberately don't
			// have.
			enableMipPyramid: false
		} );

		const result = await trainer.train( { renderer, sourceTextures: [ pack0, pack1 ] } );
		const cpuModel = result.cpuModel;
		const levelTextures = buildLevelTextures( cpuModel );

		async function queryDxDy( u, v ) {

			const activations = evaluateNeuralTextureRaw( vec2( u, v ), cpuModel, levelTextures );
			const dxPred = applyChannelActivation( activations[ 3 ], 'tanh' );
			const dyPred = applyChannelActivation( activations[ 4 ], 'tanh' );
			const colorNode = vec4( dxPred, dyPred, 0, 1 );

			const renderTarget = await bakeColorNodeToTexture( renderer, colorNode, 2 );
			const rawPixels = await renderer.readRenderTargetPixelsAsync( renderTarget, 0, 0, 1, 1 );
			renderTarget.dispose();

			// HalfFloatType target - readback is raw half-float bit patterns, see
			// NeuralAppearanceTeacherReadback.js's readPixelValue.
			return {
				dx: THREE.DataUtils.fromHalfFloat( rawPixels[ 0 ] ),
				dy: THREE.DataUtils.fromHalfFloat( rawPixels[ 1 ] )
			};

		}

		// Stripe midpoints for period 4: [0, .25) +, [.25, .5) -, [.5, .75) +,
		// [.75, 1) -. Query dx across all four u-stripes at a fixed v (itself
		// inside a "+" dy stripe, though dx/dy are independent outputs so this
		// choice shouldn't matter).
		const dxSamples = await Promise.all( [ 0.125, 0.375, 0.625, 0.875 ].map( ( u ) => queryDxDy( u, 0.125 ) ) );
		const dySamples = await Promise.all( [ 0.125, 0.375, 0.625, 0.875 ].map( ( v ) => queryDxDy( 0.125, v ) ) );

		const expectedSigns = [ 1, - 1, 1, - 1 ];

		for ( let i = 0; i < 4; i ++ ) {

			const expected = expectedSigns[ i ];
			// Sign must match...
			expect( Math.sign( dxSamples[ i ].dx ), `dx stripe ${i} sign` ).toBe( expected );
			// ...and with real amplitude, not a near-zero value that technically
			// has the right sign but reads as "no bump" visually (the actual
			// symptom being chased).
			expect( Math.abs( dxSamples[ i ].dx ), `dx stripe ${i} magnitude` ).toBeGreaterThan( 0.15 );

			expect( Math.sign( dySamples[ i ].dy ), `dy stripe ${i} sign` ).toBe( expected );
			expect( Math.abs( dySamples[ i ].dy ), `dy stripe ${i} magnitude` ).toBeGreaterThan( 0.15 );

		}

		for ( const levelTexture of levelTextures ) levelTexture.dispose();
		pack0.dispose();
		pack1.dispose();

	}, 120000 );

} );

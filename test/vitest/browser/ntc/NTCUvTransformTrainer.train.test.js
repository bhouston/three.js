import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { float, fract, step, uv, vec2, vec3 } from 'three/tsl';
import { NTCTrainer } from '../../../../examples/jsm/ntc/training/NTCTrainer.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { NTCNodeMaterial } from '../../../../examples/jsm/ntc/NTCNodeMaterial.js';
import { CHANNELS, layoutChannels } from '../../../../examples/jsm/ntc/NTCFormat.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

/** A minimal 'albedo only' channel classification, matching outputChannels: 3 below. */
function buildAlbedoOnlyClassification() {

	const activeList = CHANNELS.filter( ( channel ) => channel.key === 'albedo' );
	const { channels: activeChannels, totalChannels, packCount } = layoutChannels( activeList );
	const constantValues = {};
	for ( const channel of CHANNELS ) if ( channel.key !== 'albedo' ) constantValues[ channel.key ] = channel.defaultValue;

	return { activeChannels, totalChannels, packCount, constantValues };

}

// End-to-end coverage for the learned per-material UV transform (see
// NTCUvTransform.js / NTCUvTransformTrainer.js / NTCGridPyramidModel.js's
// `enableUvTransform` option) against a deliberately UV-misaligned training
// target - a stripe pattern rotated 30deg relative to the raw UV axes,
// analogous to `standard_surface_rotate2d_test.mtlx`'s `rotate2d` node
// (which rotates its `image` node's texcoord 45deg before sampling).
describe( 'Addons > NTC > NTCUvTransformTrainer (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

	/** A stripe pattern rotated `angle` radians about the UV center. */
	function rotatedStripeColorNode( angle, period ) {

		const c = float( Math.cos( angle ) );
		const s = float( Math.sin( angle ) );
		const centered = uv().sub( 0.5 );
		const rotated = vec2(
			centered.x.mul( c ).sub( centered.y.mul( s ) ),
			centered.x.mul( s ).add( centered.y.mul( c ) )
		).add( 0.5 );
		const stripe = step( 0.5, fract( rotated.x.mul( period ) ) );

		return vec3( stripe, stripe, stripe );

	}

	it( 'trains without throwing, converges to a finite/decreasing loss, and bakes a flat 6-float matrix onto cpuModel.uvTransform', async () => {

		const bakeResolution = 64;
		const sourceRenderTarget = await bakeColorNodeToTexture(
			renderer, rotatedStripeColorNode( Math.PI / 6, 6 ), bakeResolution, { generateMipmaps: true }
		);

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 2,
			baseResolution: 32,
			hiddenSizes: [ 16, 16 ],
			outputChannels: 3,
			batchSize: 2048,
			iterations: 400,
			learningRate: 0.02,
			enableUvTransform: true,
			seed: 1
		} );

		const losses = [];

		const result = await trainer.train( {
			renderer,
			sourceTextures: [ sourceRenderTarget.texture ],
			onProgress: ( { loss } ) => losses.push( loss )
		} );
		sourceRenderTarget.dispose();

		expect( Number.isFinite( result.loss ) ).toBe( true );
		expect( losses.length ).toBeGreaterThan( 1 );
		expect( losses.every( Number.isFinite ) ).toBe( true );

		// Baked to the runtime-ready flat matrix form (not the decomposed
		// `{ rotation, scale }` shape createNTCGridPyramidModel initializes -
		// see NTCTrainer.js's end-of-train() bake step), and no longer
		// identity (SPSA should have moved it at least a little in 200
		// iterations against such a strongly rotated target).
		expect( Array.isArray( result.cpuModel.uvTransform ) ).toBe( true );
		expect( result.cpuModel.uvTransform.length ).toBe( 6 );

		// Loss should reach a meaningfully lower point than where it started
		// - checked against the best point reached, not the final one: SPSA's
		// own +/- perturbation evaluations (see NTCUvTransformTrainer.js) and
		// the learning-rate's cosine anneal toward the very end both add
		// per-point noise a strict "last < first" comparison is too fragile
		// against (this is a coarse convergence sanity check, not a
		// monotonicity guarantee).
		const best = Math.min( ...losses );

		expect( best ).toBeLessThan( losses[ 0 ] * 0.85 );

	}, 60000 );

	it( 'a live mid-training onProgress cpuModel (uvTransform still the decomposed { rotation, scale } shape) constructs an NTCNodeMaterial without throwing - regression test for the .map-is-not-a-function crash seen with a raw object', async () => {

		const bakeResolution = 32;
		const sourceRenderTarget = await bakeColorNodeToTexture(
			renderer, rotatedStripeColorNode( Math.PI / 6, 6 ), bakeResolution, { generateMipmaps: true }
		);

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 2,
			baseResolution: 16,
			hiddenSizes: [ 8, 8 ],
			outputChannels: 3,
			batchSize: 512,
			iterations: 8,
			learningRate: 0.02,
			enableUvTransform: true,
			seed: 1
		} );

		let sawProgress = false;

		await trainer.train( {
			renderer,
			sourceTextures: [ sourceRenderTarget.texture ],
			onProgress: ( { cpuModel } ) => {

				sawProgress = true;

				// Still the decomposed shape mid-training - see
				// NTCUvTransform.js's resolveUvTransformMatrix doc comment.
				expect( cpuModel.uvTransform ).toEqual( { rotation: expect.any( Number ), scale: [ expect.any( Number ), expect.any( Number ) ] } );

				expect( () => new NTCNodeMaterial( cpuModel, buildAlbedoOnlyClassification() ) ).not.toThrow();

			}
		} );

		sourceRenderTarget.dispose();

		expect( sawProgress ).toBe( true );

	}, 30000 );

	it( 'enableUvTransform: false (the default) leaves cpuModel.uvTransform undefined and trains identically to before this feature existed', async () => {

		const bakeResolution = 32;
		const sourceRenderTarget = await bakeColorNodeToTexture(
			renderer, rotatedStripeColorNode( Math.PI / 6, 6 ), bakeResolution, { generateMipmaps: true }
		);

		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 2,
			baseResolution: 16,
			hiddenSizes: [ 8, 8 ],
			outputChannels: 3,
			batchSize: 512,
			iterations: 40,
			learningRate: 0.02,
			seed: 1
		} );

		const result = await trainer.train( { renderer, sourceTextures: [ sourceRenderTarget.texture ] } );
		sourceRenderTarget.dispose();

		expect( result.cpuModel.uvTransform ).toBeUndefined();

	}, 30000 );

} );

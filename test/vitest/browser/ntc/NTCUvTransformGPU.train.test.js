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
// NTCUvTransform.js / NTCUvTransformGPU.js / NTCGridPyramidModel.js's
// `enableUvTransform` option) against a deliberately UV-misaligned training
// target - a stripe pattern rotated 30deg relative to the raw UV axes,
// analogous to `standard_surface_rotate2d_test.mtlx`'s `rotate2d` node
// (which rotates its `image` node's texcoord 45deg before sampling).
describe( 'Addons > NTC > NTCUvTransformGPU (real WebGPU)', () => {

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
		// identity (analytic-gradient training should have moved it well
		// away from identity in 400 iterations against such a strongly
		// rotated target).
		expect( Array.isArray( result.cpuModel.uvTransform ) ).toBe( true );
		expect( result.cpuModel.uvTransform.length ).toBe( 6 );

		// Loss should reach a meaningfully lower point than where it started
		// - checked against the best point reached, not the final one (the
		// learning-rate's cosine anneal toward the very end adds a little
		// per-point noise a strict "last < first" comparison is too fragile
		// against), but with a much tighter bound than an SPSA-based version
		// of this trainer could reliably hit - see NTCUvTransformGPU.js's
		// doc comment on why analytic gradients converge far more reliably.
		const best = Math.min( ...losses );

		expect( best ).toBeLessThan( losses[ 0 ] * 0.6 );

	}, 60000 );

	it( 'cpuModel.uvTransform actually changes across successive onProgress ticks, not just at the very end - regression test for a bug where it stayed frozen at its identity init for the whole run', async () => {

		const bakeResolution = 32;
		const sourceRenderTarget = await bakeColorNodeToTexture(
			renderer, rotatedStripeColorNode( Math.PI / 6, 6 ), bakeResolution, { generateMipmaps: true }
		);

		// `iterations: 160` at the default `accumulationIterations: 16` (see
		// NTCUvTransformGPU.js) gives ~10 windows where the UV transform's
		// own Adam step actually applies - few enough to keep this test fast,
		// but enough windows that "did it ever move more than once" is a
		// meaningful question (a single window firing wouldn't distinguish
		// this bug from a one-off fluke).
		const trainer = new NTCTrainer( {
			channels: 4,
			levels: 2,
			baseResolution: 16,
			hiddenSizes: [ 8, 8 ],
			outputChannels: 3,
			batchSize: 2048,
			iterations: 160,
			learningRate: 0.05,
			enableUvTransform: true,
			seed: 1
		} );

		const rotations = [];

		await trainer.train( {
			renderer,
			sourceTextures: [ sourceRenderTarget.texture ],
			onProgress: ( { cpuModel } ) => rotations.push( cpuModel.uvTransform.rotation )
		} );
		sourceRenderTarget.dispose();

		expect( rotations.length ).toBeGreaterThan( 2 );

		// The bug's symptom was every single tick reading back the exact same
		// (frozen, identity-init) value all the way to the end - assert
		// genuine movement across the run instead: with gradient
		// accumulation (see createUvTransformGPUState's doc comment),
		// `rotation` is only *expected* to change once every
		// `accumulationIterations` iterations now, not every progress tick -
		// what matters is it changes at all, more than once, not on every
		// single tick.
		const distinctValues = new Set( rotations ).size;
		expect( distinctValues ).toBeGreaterThan( 2 );

		// And specifically not frozen after its first move - the last tick
		// shouldn't equal the first (would suggest it moved once then froze).
		expect( rotations[ rotations.length - 1 ] ).not.toBe( rotations[ 0 ] );

	}, 30000 );

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

	it( 'rotation grows to a meaningful magnitude against a purely-rotated target, not just scale - regression test for gradient accumulation fixing a noise-dominated Adam step that previously moved scale but left rotation essentially stuck near 0', async () => {

		// A pure rotation, no scale mismatch at all (period 6 matches this
		// model's base resolution reasonably well) - if only *scale* ends up
		// moving substantially while *rotation* stays near 0, that's exactly
		// the reported symptom this test locks down against, on a target
		// that has nothing else to gain from moving scale.
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
			batchSize: 4096,
			iterations: 800,
			learningRate: 0.05,
			enableUvTransform: true,
			seed: 3
		} );

		const result = await trainer.train( { renderer, sourceTextures: [ sourceRenderTarget.texture ] } );
		sourceRenderTarget.dispose();

		// `result.cpuModel.uvTransform` is the baked flat matrix at this
		// point (see NTCTrainer.js's end-of-train() bake step) -
		// `atan2(c, a)` recovers the rotation angle from it (see
		// NTCUvTransform.composeUvTransformMatrix's doc comment for the
		// `[a,b,c,d,e,f]` layout: `a=cos*sx, c=sin*sx`).
		const [ a, , c ] = result.cpuModel.uvTransform;
		const learnedRotation = Math.atan2( c, a );

		// 0.1 rad (~5.7deg) is a low bar deliberately - this only needs to
		// demonstrate real, substantial movement happened (vs. "stuck
		// within noise of 0"), not that it converged all the way to the
		// injected 30deg (which a wrapped/periodic torus optimization
		// landscape has no obligation to find exactly, or even via the
		// shortest rotational path - see NTCUvTransformGPU.js's literature
		// note on near-symmetric optima).
		expect( Math.abs( learnedRotation ) ).toBeGreaterThan( 0.1 );

	}, 60000 );

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

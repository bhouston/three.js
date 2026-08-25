import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { float, fract, step, uv, vec2, vec3 } from 'three/tsl';
import { NTCGPUModel } from '../../../../examples/jsm/ntc/training/NTCGPUModel.js';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';
import { createTextureTrainBatchComputeNode } from '../../../../examples/jsm/ntc/training/NTCGPUComputeTSL.js';
import {
	ROTATION_INDEX,
	LOG_SCALE_X_INDEX,
	LOG_SCALE_Y_INDEX,
	createUvTransformGPUState,
	disposeUvTransformGPUState
} from '../../../../examples/jsm/ntc/training/NTCUvTransformGPU.js';
import { FIXED_POINT_SCALE } from '../../../../examples/jsm/ntc/training/NTCGPUTrainingConstants.js';
import { createRandom } from '../../../../examples/jsm/ntc/training/NTCTrainingUtils.js';
import { bakeColorNodeToTexture } from '../../../../examples/jsm/ntc/training/NTCTextureSource.js';
import { createTestRenderer } from '../helpers/webgpuEval.js';

// Isolates "is the analytic UV-transform gradient (NTCGPUComputeTSL.js's
// step '5.5') actually correct" from "is Adam/learning-rate/tuning good" (see
// NTCUvTransformGPU.train.test.js for the latter, end-to-end concern) -
// directly compares the kernel's own accumulated `dL/d(rotation, log-scale)`
// against a finite-difference estimate of the *same* batch's loss, computed
// by perturbing each parameter and re-running the identical forward+backward
// pass (kept identical by fixing `stepUniform`, which is what the random
// UV/LOD draws are seeded from - see NTCGPUComputeTSL.randomStratifiedUV/
// sampleTrainingLod).
//
// Averaged over several independent batches (`SEEDS` below), not a single
// one: an earlier single-batch version of this test was itself unreliable -
// flipping which comparisons passed/failed depending on `eps` alone, with no
// code change - because a *single* 512-sample batch's gradient for these 3
// heavily-pooled global scalars is dominated by which handful of samples
// happen to land near a texel/level-selection boundary (where the loss isn't
// locally smooth), not by the true underlying signal. This single fact is
// also the actual explanation for what motivated writing this test in the
// first place (see NTCTrainer.js's `uvTransformAccumulationIterations`):
// Adam applied to that same single-batch-noisy gradient *every* iteration
// (as weights/latents' own Adam step safely is - but they have millions of
// independently-initialized parameters for the law of large numbers to
// smooth over, not 3 shared global ones) reads as a UV transform that drifts
// substantially even on content with nothing to gain from moving at all.
describe( 'Addons > NTC > NTCUvTransformGPU analytic gradient vs. finite difference (real WebGPU)', () => {

	let renderer;

	beforeAll( async () => {

		renderer = await createTestRenderer();

	} );

	afterAll( () => {

		renderer?.dispose();
		renderer = undefined;

	} );

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

	it( 'dL/d(rotation), dL/d(logScaleX) and dL/d(logScaleY) all match a central finite-difference estimate in sign and rough magnitude', async () => {

		const bakeResolution = 32;
		const sourceRenderTarget = await bakeColorNodeToTexture(
			renderer, rotatedStripeColorNode( Math.PI / 6, 6 ), bakeResolution, { generateMipmaps: true }
		);

		const modelSettings = {
			channels: 4, levels: 2, baseResolution: 16, hiddenSizes: [ 8, 8 ],
			outputChannels: 3, batchSize: 512, textureResolution: bakeResolution
		};

		// A real (non-degenerate) seeded random - NOT the constant-0.5
		// convenience some other tests use for "all-zero weights", which
		// would make gradA0 (and therefore every UV-transform gradient
		// below) identically zero regardless of any actual bug, since
		// gradA0 is a weighted sum by decoder weights that would all be 0.
		const cpuModel = createNTCGridPyramidModel( modelSettings, createRandom( 42 ) );
		const gpuModel = new NTCGPUModel( modelSettings );
		gpuModel.initFromCPUModel( cpuModel );

		const uvTransformState = createUvTransformGPUState();
		const trainBatchNode = createTextureTrainBatchComputeNode(
			gpuModel, [ sourceRenderTarget.texture ], uvTransformState, 0
		);

		async function evalAtSeed( rotation, logScaleX, logScaleY, seed ) {

			gpuModel.stepUniform.value = seed;

			const values = uvTransformState.buffers.attribute.array;
			values[ ROTATION_INDEX ] = rotation;
			values[ LOG_SCALE_X_INDEX ] = logScaleX;
			values[ LOG_SCALE_Y_INDEX ] = logScaleY;
			uvTransformState.buffers.attribute.needsUpdate = true;

			// No Adam step runs in this test (which is normally what zeroes
			// the gradient accumulator between iterations, via its own
			// atomicStore - see NTCGPUKernelsTSL.createAdamComputeNode) - zero
			// it by hand so each call's readback is that call's own
			// contribution only, not an additive pile-up across calls.
			uvTransformState.buffers.gradAttribute.array.fill( 0 );
			uvTransformState.buffers.gradAttribute.needsUpdate = true;

			gpuModel.resetLoss();
			renderer.compute( trainBatchNode );

			const [ loss, gradBuffer ] = await Promise.all( [
				gpuModel.readLoss( renderer ),
				renderer.getArrayBufferAsync( uvTransformState.buffers.gradAttribute )
			] );

			const gradRaw = new Int32Array( gradBuffer );
			const toAvgGrad = ( i ) => gradRaw[ i ] / FIXED_POINT_SCALE / gpuModel.batchSize;

			return {
				loss,
				gradRotation: toAvgGrad( ROTATION_INDEX ),
				gradLogScaleX: toAvgGrad( LOG_SCALE_X_INDEX ),
				gradLogScaleY: toAvgGrad( LOG_SCALE_Y_INDEX )
			};

		}

		// Averaged over several independent batches (different `stepUniform`
		// seeds - see randomStratifiedUV/sampleTrainingLod, both derived from
		// it) to separate genuine per-batch statistical noise (a handful of
		// the 512 samples landing near a texel/level-selection boundary,
		// where the loss isn't locally smooth) from an actual formula bug -
		// a single-batch comparison, it turns out, is dominated by that noise
		// for these 3 parameters (see this test's earlier single-seed
		// version, which flipped sign on `eps` alone).
		const SEEDS = [ 1, 2, 3, 4, 5, 6, 7, 8 ];

		async function evalAt( rotation, logScaleX, logScaleY ) {

			const samples = [];
			for ( const seed of SEEDS ) samples.push( await evalAtSeed( rotation, logScaleX, logScaleY, seed ) );

			const avg = ( key ) => samples.reduce( ( a, s ) => a + s[ key ], 0 ) / samples.length;

			return {
				loss: avg( 'loss' ),
				gradRotation: avg( 'gradRotation' ),
				gradLogScaleX: avg( 'gradLogScaleX' ),
				gradLogScaleY: avg( 'gradLogScaleY' )
			};

		}

		// Evaluated away from identity (not rotation=0, scale=1) so every
		// term in the Jacobian (sin AND cos, both scale axes) is actually
		// exercised, not accidentally zeroed by a special-case value.
		const base = { rotation: 0.4, logScaleX: 0.15, logScaleY: - 0.1 };
		const eps = 0.01;

		const atBase = await evalAt( base.rotation, base.logScaleX, base.logScaleY );

		const rotationPlus = await evalAt( base.rotation + eps, base.logScaleX, base.logScaleY );
		const rotationMinus = await evalAt( base.rotation - eps, base.logScaleX, base.logScaleY );
		const logSxPlus = await evalAt( base.rotation, base.logScaleX + eps, base.logScaleY );
		const logSxMinus = await evalAt( base.rotation, base.logScaleX - eps, base.logScaleY );
		const logSyPlus = await evalAt( base.rotation, base.logScaleX, base.logScaleY + eps );
		const logSyMinus = await evalAt( base.rotation, base.logScaleX, base.logScaleY - eps );

		sourceRenderTarget.dispose();
		disposeUvTransformGPUState( uvTransformState );
		gpuModel.dispose();

		const finiteDiff = {
			rotation: ( rotationPlus.loss - rotationMinus.loss ) / ( 2 * eps ),
			logScaleX: ( logSxPlus.loss - logSxMinus.loss ) / ( 2 * eps ),
			logScaleY: ( logSyPlus.loss - logSyMinus.loss ) / ( 2 * eps )
		};

		for ( const [ param, analyticKey ] of [
			[ 'rotation', 'gradRotation' ],
			[ 'logScaleX', 'gradLogScaleX' ],
			[ 'logScaleY', 'gradLogScaleY' ]
		] ) {

			const analytic = atBase[ analyticKey ];
			const fd = finiteDiff[ param ];

			// Same sign - the actual correctness question ("does the kernel
			// even point the right direction for this parameter"), robust to
			// the bilinear-tap gradient's known floor()-ignoring
			// approximation (see NTCGPUComputeTSL.js step 5.5's doc comment)
			// introducing a small quantitative (not qualitative) mismatch.
			expect( Math.sign( analytic ), `${param}: analytic=${analytic}, finiteDiff=${fd}` ).toBe( Math.sign( fd ) );

			// Same rough order of magnitude (within 3x either way) - loose on
			// purpose (this isn't meant to catch floating-point-level drift,
			// just a formula/index/sign-flip class of bug).
			const ratio = analytic / fd;
			expect( ratio, `${param}: analytic=${analytic}, finiteDiff=${fd}` ).toBeGreaterThan( 1 / 3 );
			expect( ratio, `${param}: analytic=${analytic}, finiteDiff=${fd}` ).toBeLessThan( 3 );

		}

	}, 30000 );

} );

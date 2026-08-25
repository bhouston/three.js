import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform, vec2 } from 'three/tsl';
import { createAdamParameterBuffers, disposeAdamParameterBuffers, createAdamComputeNode } from './NTCGPUKernelsTSL.js';
import { composeUvTransformMatrix, composeUvTransformMatrixTSL } from '../NTCUvTransform.js';

// GPU-resident training state for the 3 trainable UV-transform scalars
// (rotation, log-scale x, log-scale y - see NTCUvTransform.js's doc comment
// for why they're decomposed rather than a raw 6-DOF matrix) - trained by a
// real analytic gradient computed inside the same hand-differentiated
// backward pass every other parameter in this system uses (see
// NTCGPUComputeTSL.js's "5.5" step), *not* a gradient-free outer loop - see
// this module's doc comment on why that matters.
//
// Shaped identically to how NTCGPUModel.js holds weights/latents
// (createAdamParameterBuffers: value/gradient-atomic/Adam-moment buffers),
// just at a count of 3 instead of millions - the exact same
// NTCGPUKernelsTSL.createAdamComputeNode used for weights/latents applies
// here unchanged (see createUvTransformAdamComputeNode below).
//
// Literature note: every real Spatial Transformer Networks implementation
// (Jaderberg et al. 2015, and everything downstream) trains its affine
// parameters this way - backprop through the differentiable sampler - not
// with a gradient-free method. An earlier version of this feature used SPSA
// (Simultaneous Perturbation Stochastic Approximation) as a build-fast
// prototype; in practice it converged poorly (SPSA is a provably higher-
// variance, slower-converging estimator - Spall's own results put it at
// O(n^-1/3) vs. O(n^-1/2) for a real stochastic gradient - and the
// rotation/scale loss landscape has near-symmetric optima a noisy zeroth-
// order estimate can drift between instead of settling into), so this
// analytic version replaces it.

const ROTATION_INDEX = 0;
const LOG_SCALE_X_INDEX = 1;
const LOG_SCALE_Y_INDEX = 2;
const UV_TRANSFORM_PARAM_COUNT = 3;

// How many training iterations' worth of gradient accumulate (via the same
// atomicAdd every other trained parameter's gradient does, see
// NTCGPUComputeTSL.js's step "5.5") before one Adam step is actually applied
// - see createUvTransformGPUState's doc comment on why this matters. Default
// chosen empirically (see NTCUvTransformGPU.train.test.js) - large enough to
// meaningfully reduce noise, small enough that the transform still visibly
// moves within a normal few-thousand-iteration run.
const DEFAULT_ACCUMULATION_ITERATIONS = 16;

/**
 * Allocates the GPU state above and the live TSL nodes/matrix the training
 * kernel's forward pass reads every invocation (see
 * NTCGPUComputeTSL.js's `uvTransformState` parameter) - `rotationNode`/
 * `logScaleXNode`/`logScaleYNode` read straight off the live value buffer
 * (`buffers.valuesStorage`, the same buffer the Adam step updates each
 * iteration - see createUvTransformAdamComputeNode), so the forward pass
 * always sees the current, already-being-optimized transform with no extra
 * sync step needed. Initialized to `[0, 0, 0]` (a `Float32Array`'s default
 * zero-fill, needing no explicit init call) - rotation 0, `exp(0) = 1`
 * scale on both axes, i.e. the identity transform, matching every other
 * identity-initialized piece of this feature (see NTCGridPyramidModel.js's
 * doc comment on why - standard Spatial Transformer Networks practice).
 *
 * `options.batchSize` (required) and `options.accumulationIterations`
 * (default `DEFAULT_ACCUMULATION_ITERATIONS`) together size
 * `invBatchUniform`, used only by these 3 parameters' own Adam step (see
 * createUvTransformAdamComputeNode) - *not* the same `invBatchUniform`
 * weights/latents use (`gpuModel.invBatchUniform`, always `1/batchSize`,
 * since their gradient resets every single iteration). These 3 global
 * scalars' gradient is deliberately left to *accumulate* across
 * `accumulationIterations` iterations before NTCTrainer.js applies an Adam
 * step at all (skipping the intervening iterations' `createAdamComputeNode`
 * call, whose own `atomicStore(...,0)` is the only thing that ever resets
 * this buffer - so simply not calling it lets contributions pile up) -
 * `invBatchUniform = 1/(accumulationIterations*batchSize)` converts that
 * larger accumulated sum back into a proper per-sample mean.
 *
 * Why this matters, concretely (found via NTCUvTransformGPU.gradient.test.js
 * comparing this kernel's own gradient against a finite-difference estimate
 * of the same loss): the *formula* is correct, but a single iteration's
 * per-sample contribution to `dL/d(rotation, log-scale)` is only nonzero for
 * samples whose bilinear tap actually straddles a real edge in the selected
 * latent grid level - everywhere else the local grid content is close to
 * uniform, so `d(bilinear)/d(coordinate) ~= 0` regardless of loss. That's a
 * small, noisy fraction of any one batch, unlike weights/latents (which get
 * a useful gradient from every sample that lands near *their own* latent
 * texel). Applying Adam - which reacts with a roughly learning-rate-sized
 * step whenever there's *any* per-parameter signal at all - to that
 * single-batch noise every iteration reads as this feature "moving a lot
 * without ever converging", including on content with nothing to gain from
 * moving. Accumulating several iterations' worth of gradient before each
 * step multiplies the effective number of edge-adjacent samples informing
 * it, directly fixing the noise this discovered rather than just damping its
 * symptom (e.g. a smaller learning rate would only slow the same random
 * walk down, not make it a less noisy estimate).
 *
 * Gradient-norm clipping (see NTCGPUKernelsTSL.computeGradientClipScale) is
 * deliberately not wired up for these 3 parameters - `gradNormAtomic` is
 * allocated but never written to, which makes `computeGradientClipScale`'s
 * clip scale permanently `1` (no clipping active). That's a deliberate
 * choice, not an oversight: unlike weights/latents (millions of
 * independently-initialized values, prone to occasional per-sample
 * outliers a shared norm clip genuinely protects against), these are 3
 * already-bounded quantities (rotation is periodic, log-scale grows slowly
 * under Adam's own bounded per-step magnitude, and now genuinely
 * noise-reduced by accumulation too) - NTCTrainer.js's end-of-training bake
 * step clamps the final scale as a last-resort safety net instead.
 */
function createUvTransformGPUState( options = {} ) {

	const buffers = createAdamParameterBuffers( UV_TRANSFORM_PARAM_COUNT );

	const gradNormAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
	const gradNormAtomic = storage( gradNormAttribute, 'int', 1 ).toAtomic();
	const maxGradientNormUniform = uniform( options.maxGradientNorm || 1000 );

	const accumulationIterations = options.accumulationIterations || DEFAULT_ACCUMULATION_ITERATIONS;
	const invBatchUniform = uniform( 1 / ( accumulationIterations * options.batchSize ) );

	const rotationNode = buffers.valuesStorage.element( ROTATION_INDEX );
	const logScaleXNode = buffers.valuesStorage.element( LOG_SCALE_X_INDEX );
	const logScaleYNode = buffers.valuesStorage.element( LOG_SCALE_Y_INDEX );

	return {
		buffers,
		gradNormAttribute,
		gradNormAtomic,
		maxGradientNormUniform,
		accumulationIterations,
		invBatchUniform,
		rotationNode,
		logScaleXNode,
		logScaleYNode,
		matrixNode: composeUvTransformMatrixTSL( rotationNode, vec2( logScaleXNode, logScaleYNode ) )
	};

}

/**
 * The Adam step for the 3 UV-transform scalars - a direct reuse of
 * `NTCGPUKernelsTSL.createAdamComputeNode`, the same builder
 * `createTextureAdamWeightsComputeNode`/`createTextureAdamLatentsComputeNode`
 * wrap for weights/latents (see NTCGPUComputeTSL.js). `learningRateUniform`/
 * `stepUniform` are meant to be the *same* uniforms the rest of that
 * iteration's training step already uses (`gpuModel`'s), so this parameter
 * set anneals on the same learning-rate schedule as everything else - but
 * `invBatchUniform` is `state`'s own (see createUvTransformGPUState's doc
 * comment on why it can't be `gpuModel.invBatchUniform`). NTCTrainer.js is
 * expected to call the returned node only once every
 * `state.accumulationIterations` iterations, not every iteration - see that
 * same doc comment.
 */
function createUvTransformAdamComputeNode( state, { learningRateUniform, stepUniform } ) {

	return createAdamComputeNode( {
		valuesStorage: state.buffers.valuesStorage,
		gradAtomic: state.buffers.gradAtomic,
		mStorage: state.buffers.mStorage,
		vStorage: state.buffers.vStorage,
		gradNormAtomic: state.gradNormAtomic,
		maxGradientNormUniform: state.maxGradientNormUniform,
		learningRateUniform,
		stepUniform,
		invBatchUniform: state.invBatchUniform,
		count: UV_TRANSFORM_PARAM_COUNT,
		name: 'NTCAdamUvTransform'
	} );

}

/**
 * Reads the 3 trained scalars back from GPU, raw (unclamped, still in
 * log-scale) - shared by `readUvTransformParams`/`readUvTransformMatrix`
 * below, and the one GPU->CPU round trip either needs.
 */
async function _readRawParams( state, renderer ) {

	const buffer = await renderer.getArrayBufferAsync( state.buffers.attribute );
	const [ rotation, logSx, logSy ] = new Float32Array( buffer );

	return { rotation, logSx, logSy };

}

/**
 * Reads the 3 trained scalars back from GPU as the decomposed `{ rotation,
 * scale }` shape `createNTCGridPyramidModel` initializes `cpuModel.
 * uvTransform` to (see NTCGridPyramidModel.js) - meant for a *periodic*
 * mid-training readback (see NTCTrainer.js's `shouldSync` block, the same
 * cadence weights/latents already sync on via `gpuModel.syncToCPU`), so an
 * `onProgress` callback - and therefore any live preview built from it, e.g.
 * `NTCNodeMaterial`'s `UV_DEBUG_VIEW` - actually reflects the current
 * transform instead of staying frozen at its identity init for the whole
 * run (only `readUvTransformMatrix`, called once at the very end, updated
 * `cpuModel.uvTransform` before this existed - a real bug, not a deliberate
 * choice: every other trained parameter syncs on this same cadence).
 * Deliberately unclamped (unlike `readUvTransformMatrix`) - this is a live
 * preview of the actual in-progress value, not a final export.
 */
async function readUvTransformParams( state, renderer ) {

	const { rotation, logSx, logSy } = await _readRawParams( state, renderer );

	return { rotation, scale: [ Math.exp( logSx ), Math.exp( logSy ) ] };

}

/**
 * Reads the 3 trained scalars back from GPU and bakes them to the flat
 * 6-float matrix form the manifest/runtime consumes (see
 * NTCUvTransform.composeUvTransformMatrix) - called once at the end of
 * training (see NTCTrainer.js). `maxLogScale` (default `Math.log(4)`, i.e.
 * a 4x scale ceiling in either direction) is the last-resort safety clamp
 * mentioned in this module's doc comment - real training shouldn't need it,
 * but bakes a sane, invertible matrix even if something did diverge, rather
 * than silently exporting an extreme or degenerate (near-zero-scale) one.
 */
async function readUvTransformMatrix( state, renderer, maxLogScale = Math.log( 4 ) ) {

	const { rotation, logSx, logSy } = await _readRawParams( state, renderer );

	const clampedLogSx = Math.min( maxLogScale, Math.max( - maxLogScale, logSx ) );
	const clampedLogSy = Math.min( maxLogScale, Math.max( - maxLogScale, logSy ) );

	return composeUvTransformMatrix( rotation, [ Math.exp( clampedLogSx ), Math.exp( clampedLogSy ) ] );

}

function disposeUvTransformGPUState( state ) {

	disposeAdamParameterBuffers( state.buffers );
	state.gradNormAttribute.dispose();

}

export {
	ROTATION_INDEX,
	LOG_SCALE_X_INDEX,
	LOG_SCALE_Y_INDEX,
	createUvTransformGPUState,
	createUvTransformAdamComputeNode,
	readUvTransformParams,
	readUvTransformMatrix,
	disposeUvTransformGPUState
};

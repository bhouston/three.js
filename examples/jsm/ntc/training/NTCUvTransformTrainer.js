import { uniform, vec2 } from 'three/tsl';
import { composeUvTransformMatrix, composeUvTransformMatrixTSL } from '../NTCUvTransform.js';

const DEFAULT_SPSA_OPTIONS = {
	// How often (in training iterations) an SPSA outer step runs - see this
	// class's doc comment for why each outer step costs 2 full training-batch
	// evaluations instead of the usual 1, so this trades UV-transform
	// convergence speed against overall training throughput.
	interval: 4,
	// SPSA's perturbation magnitude `c` (Spall 1992's notation) - applied
	// directly to rotation (radians) and log-scale alike, since both are
	// already unitless/dimensionless quantities of comparable scale.
	perturbation: 0.05,
	learningRate: 0.02,
	beta1: 0.9,
	beta2: 0.999,
	epsilon: 1e-8
};

/**
 * Gradient-free (SPSA - Simultaneous Perturbation Stochastic Approximation,
 * Spall 1992) outer-loop optimizer for the 3 global UV-transform scalars
 * (rotation, log-scale x, log-scale y - see NTCUvTransform.js's doc comment
 * for why they're decomposed rather than a raw 6-DOF matrix, and why
 * translation is left out of v1).
 *
 * Deliberately *not* a hand-differentiated GPU backward pass, unlike every
 * other trainable parameter in this system (see NTCGPUComputeTSL.js's
 * gradA0/gradient-scatter steps): with only 3 scalars, SPSA needs just two
 * extra forward-*and*-backward training-batch evaluations per outer step to
 * estimate a gradient - reusing the exact same `compute()` sequence, loss
 * readback (`gpuModel.readLoss`), and live-uniform-update pattern every other
 * per-iteration value (`stepUniform`, `learningRateUniform`) already uses, in
 * NTCTrainer.js's main loop. No new GPU kernel, no new hand-derived bilinear-
 * coordinate gradient. This is an experiment-first prototype: if it
 * validates the idea (see the research that motivated it), a jointly-
 * optimized analytic gradient folded into the existing hand-written backward
 * pass is the natural next step for production use - see this class's
 * `bakeMatrix()` for the one piece of state a production version would still
 * need (the final decomposed params, baked to the runtime matrix).
 *
 * Each outer step's two training-batch evaluations (at `params + c*Δ` and
 * `params - c*Δ`) *do* run the model's normal weight/latent gradient
 * accumulation too (there's no cheap "forward-only" variant of
 * `trainBatchNode` - see NTCGPUComputeTSL.js) - but critically, *neither*
 * evaluation applies its Adam step until after both losses are read back
 * (see `step()` below): applying an Adam update between the two evaluations
 * would mean the second evaluation's loss is measured on a network the first
 * evaluation had already improved, biasing every single gradient estimate
 * toward whichever evaluation ran second, not toward the transform parameter
 * that actually reduces loss - this is not a minor/self-cancelling effect
 * (an earlier version of this class got this wrong).
 *
 * Real caveat, not just an implementation bug: SPSA is *inherently* weaker
 * than the analytic backprop-through-the-sampler gradient every actual
 * Spatial Transformer Networks implementation uses (Jaderberg et al. 2015) -
 * it's a provably higher-variance, slower-converging estimator (Spall's own
 * results put it at O(n^-1/3) vs. O(n^-1/2) for a real stochastic gradient),
 * and the rotation/scale loss landscape has near-symmetric optima (many
 * textures look similar under a 90-degree rotation or a small scale change)
 * that a noisy zeroth-order estimate can drift between instead of settling
 * into. `bakeMatrix()`'s decomposed-params state is exactly what a follow-up
 * analytic-gradient trainer would still need (see NTCGPUComputeTSL.js's
 * gradA0/gradient-scatter steps for where the bilinear-sample coordinate
 * gradient - the missing piece - would slot in) - this class is a fast
 * prototype to validate the idea, not the intended long-term training
 * method.
 */
class NTCUvTransformTrainer {

	/**
	 * `random`, a `() => number` in `[0, 1)` (see NTCTrainingUtils.
	 * createRandom), drives the SPSA sign-vector draws below - shared with
	 * (the same instance as) the rest of this trainer run's `NTCTrainer`, so
	 * a `seed` option fully determines training end-to-end, matching every
	 * other source of randomness in this system (grid/decoder init, batch
	 * jitter). Defaults to `Math.random` only so this class remains usable
	 * standalone/in a test without wiring one up.
	 */
	constructor( options = {}, random = Math.random ) {

		this.options = { ...DEFAULT_SPSA_OPTIONS, ...options };
		this.random = random;

		// Live uniforms feeding the training kernel's `uvTransformNode` (see
		// createTextureTrainBatchComputeNode) - rebuilt into the flat 6-float
		// matrix form every invocation via composeUvTransformMatrixTSL, so
		// writing `.value` here is all a step needs to do to change what the
		// next `compute()` call trains against.
		this.rotationUniform = uniform( 0 );
		this.logScaleXUniform = uniform( 0 );
		this.logScaleYUniform = uniform( 0 );

		this.matrixNode = composeUvTransformMatrixTSL(
			this.rotationUniform,
			vec2( this.logScaleXUniform, this.logScaleYUniform )
		);

		this.params = { rotation: 0, logScale: [ 0, 0 ] };

		// Tiny per-parameter Adam moments (3 scalars each) for the outer
		// SPSA gradient estimate - trivial next to the weight/latent buffers
		// this trains alongside.
		this._m = [ 0, 0, 0 ];
		this._v = [ 0, 0, 0 ];
		this._t = 0;

	}

	_writeUniforms( rotation, logScale ) {

		this.rotationUniform.value = rotation;
		this.logScaleXUniform.value = logScale[ 0 ];
		this.logScaleYUniform.value = logScale[ 1 ];

	}

	/** Sets the live uniforms to an explicit `(rotation, logScale)` pair without touching the Adam moments - used internally to switch between the +/- SPSA evaluations. */
	setParams( rotation, logScale ) {

		this.params = { rotation, logScale };
		this._writeUniforms( rotation, logScale );

	}

	/**
	 * Runs one full training-batch evaluation (the same 5-`compute()`
	 * sequence NTCTrainer.js's main loop runs every iteration) at the given
	 * `(rotation, logScale)` and returns the resulting batch loss - shared by
	 * both SPSA evaluations below.
	 */
	async _evalAt( rotation, logScale, gpu ) {

		this.setParams( rotation, logScale );

		gpu.gpuModel.resetLoss();
		gpu.renderer.compute( gpu.trainBatchNode );
		gpu.renderer.compute( gpu.resetGradientNormNode );
		gpu.renderer.compute( gpu.accumulateGradientNormNode );
		gpu.renderer.compute( gpu.adamWeightsNode );
		gpu.renderer.compute( gpu.adamLatentsNode );

		return gpu.gpuModel.readLoss( gpu.renderer );

	}

	/**
	 * Runs one SPSA outer step - see this class's doc comment. `gpu` is
	 * `{ renderer, gpuModel, trainBatchNode, resetGradientNormNode,
	 * accumulateGradientNormNode, adamWeightsNode, adamLatentsNode }`, the
	 * same compute nodes/renderer/model NTCTrainer.js's main loop already
	 * has in scope. Leaves the live uniforms holding the updated params when
	 * it returns; returns `{ loss }`, the mean of the two evaluations' losses
	 * (a reasonable per-iteration loss estimate for progress reporting).
	 */
	async step( gpu ) {

		const { perturbation, learningRate, beta1, beta2, epsilon } = this.options;
		const base = [ this.params.rotation, this.params.logScale[ 0 ], this.params.logScale[ 1 ] ];
		const delta = base.map( () => ( Math.random() < 0.5 ? - 1 : 1 ) );

		const plus = base.map( ( p, i ) => p + perturbation * delta[ i ] );
		const minus = base.map( ( p, i ) => p - perturbation * delta[ i ] );

		const lossPlus = await this._evalAt( plus[ 0 ], [ plus[ 1 ], plus[ 2 ] ], gpu );
		const lossMinus = await this._evalAt( minus[ 0 ], [ minus[ 1 ], minus[ 2 ] ], gpu );

		this._t ++;

		const next = base.slice();

		for ( let i = 0; i < 3; i ++ ) {

			const grad = ( lossPlus - lossMinus ) / ( 2 * perturbation * delta[ i ] );

			this._m[ i ] = beta1 * this._m[ i ] + ( 1 - beta1 ) * grad;
			this._v[ i ] = beta2 * this._v[ i ] + ( 1 - beta2 ) * grad * grad;

			const mHat = this._m[ i ] / ( 1 - Math.pow( beta1, this._t ) );
			const vHat = this._v[ i ] / ( 1 - Math.pow( beta2, this._t ) );

			next[ i ] = base[ i ] - learningRate * mHat / ( Math.sqrt( vHat ) + epsilon );

		}

		this.setParams( next[ 0 ], [ next[ 1 ], next[ 2 ] ] );

		return { loss: ( lossPlus + lossMinus ) / 2 };

	}

	/**
	 * Bakes the current params to the flat 6-float matrix form the manifest/
	 * runtime expects - see NTCUvTransform.composeUvTransformMatrix - called
	 * once at the end of training (see NTCTrainer.js).
	 */
	bakeMatrix() {

		const [ logSx, logSy ] = this.params.logScale;
		return composeUvTransformMatrix( this.params.rotation, [ Math.exp( logSx ), Math.exp( logSy ) ] );

	}

}

export { NTCUvTransformTrainer, DEFAULT_SPSA_OPTIONS };

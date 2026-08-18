import { Fn, If, atomicLoad, atomicStore, float, instanceIndex, int, max, min, pow, sqrt } from 'three/tsl';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from './NeuralGPUTrainingConstants.js';

// Small WebGPU compute-kernel TSL builders shared by every neural trainer
// (texture, material, appearance): wrapping a grid index, clipping by a
// global gradient norm, the two zero-fill kernels that reset gradient
// accumulators between optimizer passes, and the Adam step itself. None of
// this depends on what the gradients are gradients *of*, only on the
// fixed-point atomic layout every trainer shares via
// `NeuralGPUTrainingConstants.js`.

function wrapIndexTSL( val, size ) {

	return val.mod( size ).add( size ).mod( size );

}

function computeGradientClipScale( gradNormAtomic, maxGradientNormUniform ) {

	const normSquared = float( atomicLoad( gradNormAtomic.element( 0 ) ) ).div( float( GRADIENT_NORM_SCALE ) );
	const unclippedScale = maxGradientNormUniform.div( sqrt( max( normSquared, float( 1e-20 ) ) ) );

	return min( float( 1.0 ), unclippedScale );

}

/**
 * Clears the scalar accumulator used by the gradient clipping pass.
 */
function createResetGradientNormComputeNode( gpuModel ) {

	const { gradNormAtomic } = gpuModel;

	return Fn( () => {

		atomicStore( gradNormAtomic.element( 0 ), int( 0 ) );

	} )().compute( 1 ).setName( 'NeuralResetGradientNorm' );

}

/**
 * Clears accumulated weight and latent gradients before a scoped optimizer pass.
 */
function createResetGradientsComputeNode( gpuModel ) {

	const {
		layout,
		gradWeightsAtomic,
		gradLatentsAtomic
	} = gpuModel;
	const dispatchCount = layout.totalWeights + layout.totalLatents;

	return Fn( () => {

		const idx = int( instanceIndex );

		If( idx.lessThan( int( layout.totalWeights ) ), () => {

			atomicStore( gradWeightsAtomic.element( idx ), int( 0 ) );

		} ).Else( () => {

			atomicStore( gradLatentsAtomic.element( idx.sub( int( layout.totalWeights ) ) ), int( 0 ) );

		} );

	} )().compute( dispatchCount ).setName( 'NeuralResetGradients' );

}

/**
 * Builds one Adam optimizer step, applied to a single fixed-point gradient
 * buffer (weights or latents, whichever `valuesStorage`/`gradAtomic`/
 * `mStorage`/`vStorage` point at). Shared by every trainer's weight- and
 * latent-update kernels: same bias-corrected moment estimates, same
 * fixed-point gradient decode, same global-norm clip. What differs between
 * call sites is which buffers it targets, whether the gradient still needs
 * dividing down to a per-sample average (`invBatchUniform`), and whether the
 * dispatch is scoped to a sub-range of the buffer (`offset`/`count`) rather
 * than the whole thing - the appearance model's staged BRDF/IBL training
 * passes need that, texture and material always update the full buffer.
 */
function createAdamComputeNode( {
	valuesStorage,
	gradAtomic,
	mStorage,
	vStorage,
	gradNormAtomic,
	maxGradientNormUniform,
	learningRateUniform,
	stepUniform,
	invBatchUniform = null,
	offset = 0,
	count,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-7,
	name
} ) {

	return Fn( () => {

		const idx = int( instanceIndex ).add( int( offset ) );
		let rawGrad = float( atomicLoad( gradAtomic.element( idx ) ) ).div( float( FIXED_POINT_SCALE ) );
		if ( invBatchUniform !== null ) rawGrad = rawGrad.mul( invBatchUniform );
		const grad = rawGrad.mul( computeGradientClipScale( gradNormAtomic, maxGradientNormUniform ) );
		atomicStore( gradAtomic.element( idx ), int( 0 ) );

		const m = mStorage.element( idx );
		const v = vStorage.element( idx );
		const value = valuesStorage.element( idx );

		const nextM = float( beta1 ).mul( m ).add( float( 1.0 - beta1 ).mul( grad ) );
		const nextV = float( beta2 ).mul( v ).add( float( 1.0 - beta2 ).mul( grad ).mul( grad ) );
		mStorage.element( idx ).assign( nextM );
		vStorage.element( idx ).assign( nextV );

		const beta1Corr = float( 1.0 ).sub( pow( float( beta1 ), float( stepUniform ) ) );
		const beta2Corr = float( 1.0 ).sub( pow( float( beta2 ), float( stepUniform ) ) );
		const mHat = nextM.div( max( beta1Corr, float( 1e-10 ) ) );
		const vHat = nextV.div( max( beta2Corr, float( 1e-10 ) ) );

		const stepVal = learningRateUniform.mul( mHat ).div( sqrt( max( vHat, float( 0.0 ) ) ).add( float( epsilon ) ) );
		valuesStorage.element( idx ).assign( value.sub( stepVal ) );

	} )().compute( count ).setName( name );

}

export {
	wrapIndexTSL,
	computeGradientClipScale,
	createResetGradientNormComputeNode,
	createResetGradientsComputeNode,
	createAdamComputeNode
};

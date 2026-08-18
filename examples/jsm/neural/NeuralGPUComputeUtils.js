import { Fn, If, atomicLoad, atomicStore, float, instanceIndex, int, max, min, sqrt } from 'three/tsl';
import { GRADIENT_NORM_SCALE } from './NeuralGPUTrainingConstants.js';

// Small WebGPU compute helpers shared by every neural trainer (texture,
// material, appearance): wrapping a grid index, clipping by a global
// gradient norm, and the two zero-fill kernels that reset gradient
// accumulators between optimizer passes. None of this depends on what the
// gradients are gradients *of*, only on the fixed-point atomic layout every
// trainer shares via `NeuralGPUTrainingConstants.js`.

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

export {
	wrapIndexTSL,
	computeGradientClipScale,
	createResetGradientNormComputeNode,
	createResetGradientsComputeNode
};

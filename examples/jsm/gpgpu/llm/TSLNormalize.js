import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, Loop, float, instanceIndex, inversesqrt, storage, uint } from 'three/tsl';

/**
 * Layer normalization for a single hidden vector.
 *
 * @three_import import { TSLNormalize } from 'three/addons/gpgpu/llm/TSLNormalize.js';
 */
class TSLNormalize {

	constructor( inputNode, weightArray, biasArray, hiddenSize, options = {} ) {

		this.inputNode = inputNode;
		this.hiddenSize = hiddenSize;
		this.epsilon = options.epsilon || 1e-5;
		this.workgroupSize = options.workgroupSize || 64;

		this.weightAttribute = new StorageBufferAttribute( weightArray, 1 );
		this.biasAttribute = new StorageBufferAttribute( biasArray, 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( hiddenSize ), 1 );

		this.weightNode = storage( this.weightAttribute, 'float', hiddenSize ).toReadOnly().setName( options.name ? `${ options.name }Weight` : 'LLMLayerNormWeight' );
		this.biasNode = storage( this.biasAttribute, 'float', hiddenSize ).toReadOnly().setName( options.name ? `${ options.name }Bias` : 'LLMLayerNormBias' );
		this.outputNode = storage( this.outputAttribute, 'float', hiddenSize ).setName( options.name ? `${ options.name }Output` : 'LLMLayerNormOutput' );

		this.computeNode = this.createComputeNode( options.name || 'LLMLayerNorm' );

	}

	createComputeNode( name ) {

		const { inputNode, weightNode, biasNode, outputNode, hiddenSize, epsilon, workgroupSize } = this;

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );
			const mean = float( 0 ).toVar( 'mean' );

			Loop( { start: uint( 0 ), end: uint( hiddenSize ), type: 'uint', condition: '<' }, ( { i } ) => {

				mean.addAssign( inputNode.element( i ) );

			} );

			mean.divAssign( float( hiddenSize ) );

			const variance = float( 0 ).toVar( 'variance' );

			Loop( { start: uint( 0 ), end: uint( hiddenSize ), type: 'uint', condition: '<' }, ( { i } ) => {

				const delta = inputNode.element( i ).sub( mean );
				variance.addAssign( delta.mul( delta ) );

			} );

			variance.divAssign( float( hiddenSize ) );

			const value = inputNode.element( index ).sub( mean )
				.mul( inversesqrt( variance.add( epsilon ) ) )
				.mul( weightNode.element( index ) )
				.add( biasNode.element( index ) );

			outputNode.element( index ).assign( value );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	compute( renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLNormalize };

import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, instanceIndex, storage, uint } from 'three/tsl';

/**
 * Split packed `[q_h, gate_h]` heads into separate query and gate vectors.
 *
 * @three_import import { TSLSplitHeadGate } from 'three/addons/gpgpu/llm/TSLSplitHeadGate.js';
 */
class TSLSplitHeadGate {

	constructor( packedNode, headCount, headDim, options = {} ) {

		this.qSize = headCount * headDim;
		this.queryAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );
		this.gateAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );
		this.queryNode = storage( this.queryAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Query` : 'LLMSplitQuery' );
		this.gateNode = storage( this.gateAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Gate` : 'LLMSplitGate' );

		const packedWidth = headDim * 2;

		this.computeNode = Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( this.qSize ) ), () => {

				const head = index.div( uint( headDim ) );
				const local = index.mod( uint( headDim ) );
				const packedOffset = head.mul( uint( packedWidth ) );

				this.queryNode.element( index ).assign( packedNode.element( packedOffset.add( local ) ) );
				this.gateNode.element( index ).assign( packedNode.element( packedOffset.add( uint( headDim ) ).add( local ) ) );

			} );

		} )().compute( this.qSize, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMSplitHeadGate' );

	}

	compute( renderer ) {

		renderer.compute( this.computeNode );
		return this.queryNode;

	}

}

export { TSLSplitHeadGate };

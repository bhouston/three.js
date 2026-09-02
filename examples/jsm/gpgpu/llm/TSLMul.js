import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, instanceIndex, storage } from 'three/tsl';

/**
 * Element-wise multiply for gated MLPs.
 *
 * @three_import import { TSLMul } from 'three/addons/gpgpu/llm/TSLMul.js';
 */
class TSLMul {

	constructor( aNode, bNode, size, options = {} ) {

		this.outputAttribute = new StorageBufferAttribute( new Float32Array( size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', size ).setName( options.name ? `${ options.name }Output` : 'LLMMulOutput' );

		this.computeNode = Fn( () => {

			this.outputNode.element( instanceIndex ).assign( aNode.element( instanceIndex ).mul( bNode.element( instanceIndex ) ) );

		} )().compute( size, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMMul' );

	}

	compute( renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLMul };

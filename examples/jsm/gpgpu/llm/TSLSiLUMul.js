import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, exp, float, instanceIndex, storage } from 'three/tsl';

/**
 * Element-wise `silu(gate) * up` used by SwiGLU MLPs.
 *
 * @three_import import { TSLSiLUMul } from 'three/addons/gpgpu/llm/TSLSiLUMul.js';
 */
class TSLSiLUMul {

	constructor( gateNode, upNode, size, options = {} ) {

		this.outputAttribute = new StorageBufferAttribute( new Float32Array( size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', size ).setName( options.name ? `${ options.name }Output` : 'LLMSiLUMulOutput' );

		this.computeNode = Fn( () => {

			const x = gateNode.element( instanceIndex );
			const silu = x.div( float( 1 ).add( exp( x.negate() ) ) );

			this.outputNode.element( instanceIndex ).assign( silu.mul( upNode.element( instanceIndex ) ) );

		} )().compute( size, [ options.workgroupSize || 64 ] ).setName( options.name || 'LLMSiLUMul' );

	}

	compute( renderer ) {

		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLSiLUMul };

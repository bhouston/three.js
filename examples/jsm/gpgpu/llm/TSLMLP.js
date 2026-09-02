import { TSLGELU } from './TSLGELU.js';
import { TSLLinear } from './TSLLinear.js';

/**
 * GPT-2 MLP block: dense -> gelu_new -> dense.
 *
 * @three_import import { TSLMLP } from 'three/addons/gpgpu/llm/TSLMLP.js';
 */
class TSLMLP {

	constructor( inputNode, fcWeight, fcBias, projWeight, projBias, hiddenSize, innerSize, options = {} ) {

		this.fc = new TSLLinear( inputNode, fcWeight, fcBias, hiddenSize, innerSize, {
			name: options.name ? `${ options.name }FC` : 'LLMMLPFC',
			workgroupSize: options.workgroupSize
		} );

		this.gelu = new TSLGELU( this.fc.outputNode, innerSize, {
			name: options.name ? `${ options.name }GELU` : 'LLMMLPGELU',
			workgroupSize: options.workgroupSize
		} );

		this.proj = new TSLLinear( this.gelu.outputNode, projWeight, projBias, innerSize, hiddenSize, {
			name: options.name ? `${ options.name }Proj` : 'LLMMLPProj',
			workgroupSize: options.workgroupSize
		} );

		this.outputNode = this.proj.outputNode;
		this.computeNodes = [ this.fc.computeNode, this.gelu.computeNode, this.proj.computeNode ];

	}

	compute( renderer ) {

		this.fc.compute( renderer );
		this.gelu.compute( renderer );
		this.proj.compute( renderer );

		return this.outputNode;

	}

}

export { TSLMLP };

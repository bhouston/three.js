import { TSLGELU } from './TSLGELU.js';
import { TSLLinear } from './TSLLinear.js';
import { TSLMul } from './TSLMul.js';
import { TSLSiLUMul } from './TSLSiLUMul.js';

/**
 * Gated MLP block: `down(act(gate(x)) * up(x))`.
 *
 * Llama/Qwen use SiLU (SwiGLU). Gemma uses `gelu_pytorch_tanh` (GeGLU).
 *
 * @three_import import { TSLGatedMLP } from 'three/addons/gpgpu/llm/TSLGatedMLP.js';
 */
class TSLGatedMLP {

	constructor( inputNode, gateWeight, upWeight, downWeight, hiddenSize, innerSize, options = {} ) {

		this.gate = new TSLLinear( inputNode, gateWeight, null, hiddenSize, innerSize, {
			name: options.name ? `${ options.name }Gate` : 'LLMMLPGate',
			workgroupSize: options.workgroupSize
		} );
		this.up = new TSLLinear( inputNode, upWeight, null, hiddenSize, innerSize, {
			name: options.name ? `${ options.name }Up` : 'LLMMLPUp',
			workgroupSize: options.workgroupSize
		} );

		if ( options.activation === 'gelu_new' || options.activation === 'gelu_pytorch_tanh' ) {

			this.activatedGate = new TSLGELU( this.gate.outputNode, innerSize, {
				name: options.name ? `${ options.name }GELU` : 'LLMMLPGELU',
				workgroupSize: options.workgroupSize
			} );
			this.hidden = new TSLMul( this.activatedGate.outputNode, this.up.outputNode, innerSize, {
				name: options.name ? `${ options.name }Mul` : 'LLMMLPMul',
				workgroupSize: options.workgroupSize
			} );
			this._geluThenMul = true;

		} else {

			this.hidden = new TSLSiLUMul( this.gate.outputNode, this.up.outputNode, innerSize, {
				name: options.name ? `${ options.name }SiLUMul` : 'LLMMLPSiLUMul',
				workgroupSize: options.workgroupSize
			} );
			this._geluThenMul = false;

		}

		this.down = new TSLLinear( this.hidden.outputNode, downWeight, null, innerSize, hiddenSize, {
			name: options.name ? `${ options.name }Down` : 'LLMMLPDown',
			workgroupSize: options.workgroupSize
		} );
		this.outputNode = this.down.outputNode;
		this.computeNodes = [
			this.gate.computeNode,
			this.up.computeNode,
			...( this._geluThenMul ? [ this.activatedGate.computeNode ] : [] ),
			this.hidden.computeNode,
			this.down.computeNode
		];

	}

	compute( renderer ) {

		this.gate.compute( renderer );
		this.up.compute( renderer );

		if ( this._geluThenMul ) {

			this.activatedGate.compute( renderer );

		}

		this.hidden.compute( renderer );
		this.down.compute( renderer );

		return this.outputNode;

	}

}

export { TSLGatedMLP };

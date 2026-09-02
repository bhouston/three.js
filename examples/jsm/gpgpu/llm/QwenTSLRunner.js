import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { generateAsync } from './LLMGenerate.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { TSLConcat } from './TSLConcat.js';
import { orderedComputeNodes } from './TSLCompute.js';
import { TSLGatedDeltaNet } from './TSLGatedDeltaNet.js';
import { TSLGatedMLP } from './TSLGatedMLP.js';
import { TSLLinear } from './TSLLinear.js';
import { createChunkedLogitLayers, readChunkedLogits } from './TSLLogits.js';
import { TSLRMSNorm } from './TSLRMSNorm.js';
import { TSLSplitHeadGate } from './TSLSplitHeadGate.js';
import { QwenWeights } from './QwenWeights.js';

/**
 * Qwen3.5 text generation runner backed by TSL compute kernels.
 *
 * @three_import import { QwenTSLRunner } from 'three/addons/gpgpu/llm/QwenTSLRunner.js';
 */
class QwenTSLRunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( 'QwenEmbedding' );
		this.layers = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const name = `QwenLayer${ i }`;
			const ln1 = new TSLRMSNorm( currentNode, block.ln1Weight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }LN1`,
				workgroupSize: this.workgroupSize
			} );

			let mixer;

			if ( block.layerType === 'linear_attention' ) {

				mixer = new TSLGatedDeltaNet( ln1.outputNode, block.delta, {
					name: `${ name }Delta`,
					hiddenSize: this.hiddenSize,
					numKHeads: weights.linearKeyHeads,
					numVHeads: weights.linearValueHeads,
					keyDim: weights.linearKeyDim,
					valueDim: weights.linearValueDim,
					kernelSize: weights.linearConvKernel,
					epsilon: weights.rmsNormEps,
					workgroupSize: this.workgroupSize
				} );

			} else {

				const qGate = new TSLLinear( ln1.outputNode, block.qGateWeight, null, this.hiddenSize, weights.qSize * 2, {
					name: `${ name }QGate`,
					workgroupSize: this.workgroupSize
				} );
				const split = new TSLSplitHeadGate( qGate.outputNode, weights.headCount, weights.headDim, {
					name: `${ name }Split`,
					workgroupSize: this.workgroupSize
				} );
				const kv = new TSLLinear( ln1.outputNode, block.attnQKVWeight, null, this.hiddenSize, 2 * weights.kvSize, {
					name: `${ name }KV`,
					workgroupSize: this.workgroupSize
				} );
				const packed = new TSLConcat( [
					{ node: split.queryNode, size: weights.qSize },
					{ node: kv.outputNode, size: 2 * weights.kvSize }
				], {
					name: `${ name }Pack`,
					workgroupSize: this.workgroupSize
				} );
				const attention = new TSLAttention( packed.outputNode, this.hiddenSize, weights.headCount, this.maxTokens, {
					name: `${ name }Attention`,
					workgroupSize: this.workgroupSize,
					headDim: weights.headDim,
					kvHeadCount: weights.kvHeadCount,
					ropeTheta: weights.ropeTheta,
					rotaryDim: weights.rotaryDim,
					attnScale: weights.attnScale,
					qNormWeight: block.qNormWeight,
					kNormWeight: block.kNormWeight,
					rmsEpsilon: weights.rmsNormEps,
					offsetRMSNorm: true,
					gateNode: split.gateNode
				} );
				const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, null, weights.qSize, this.hiddenSize, {
					name: `${ name }AttnProj`,
					workgroupSize: this.workgroupSize
				} );
				mixer = {
					qGate,
					split,
					kv,
					packed,
					attention,
					attnProj,
					outputNode: attnProj.outputNode,
					computeNodes: orderedComputeNodes( qGate, split, kv, packed, attention, attnProj ),
					compute: ( renderer, position ) => {

						qGate.compute( renderer );
						split.compute( renderer );
						kv.compute( renderer );
						packed.compute( renderer );
						attention.compute( renderer, position );
						attnProj.compute( renderer );

					}
				};

			}

			const addAttention = new TSLAdd( currentNode, mixer.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const ln2 = new TSLRMSNorm( addAttention.outputNode, block.ln2Weight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }LN2`,
				workgroupSize: this.workgroupSize
			} );
			const mlp = new TSLGatedMLP( ln2.outputNode, block.mlpGateWeight, block.mlpUpWeight, block.mlpDownWeight, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: weights.mlpActivation
			} );
			const addMLP = new TSLAdd( addAttention.outputNode, mlp.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			this.layers.push( { ln1, mixer, addAttention, ln2, mlp, addMLP, layerType: block.layerType } );
			currentNode = addMLP.outputNode;

		}

		this.finalNorm = new TSLRMSNorm( currentNode, weights.outputNormWeight, this.hiddenSize, {
			epsilon: weights.rmsNormEps,
			offsetWeight: true,
			name: 'QwenFinalNorm',
			workgroupSize: this.workgroupSize
		} );
		this.logits = createChunkedLogitLayers( this.finalNorm.outputNode, weights, this.logitChunkSize, 'QwenLogits' );
		this.computeNodes = [];

		for ( const layer of this.layers ) {

			this.computeNodes.push( ...orderedComputeNodes(
				layer.ln1, layer.mixer, layer.addAttention, layer.ln2, layer.mlp, layer.addMLP
			) );

		}

		this.prefillComputeNodes = this.computeNodes.slice();
		this.computeNodes.push( ...orderedComputeNodes( this.finalNorm, ...this.logits.map( ( logit ) => logit.layer ) ) );

	}

	static async fromURL( baseURL, options = {} ) {

		return new this( await QwenWeights.fromURL( baseURL, options ), options );

	}

	computeToken( renderer, tokenId, position, computeLogits = true ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;

		for ( const layer of this.layers ) {

			if ( layer.layerType !== 'linear_attention' ) layer.mixer.attention.setPosition( position );

		}

		renderer.compute( computeLogits ? this.computeNodes : this.prefillComputeNodes );

	}

	async readLogits( renderer ) {

		return readChunkedLogits( renderer, this.logits, this.weights.vocabSize );

	}

	resetCaches() {

		for ( const layer of this.layers ) {

			if ( layer.layerType === 'linear_attention' ) layer.mixer.reset();
			else layer.mixer.attention.reset();

		}

	}

	resetCache() {

		this._cacheTokens = [];
		this._cacheLogits = null;
		this.resetCaches();

	}

	async generate( renderer, prompt, options = {} ) {

		return generateAsync( this, prompt, options, {
			rewindable: false,
			resetCache: () => this.resetCache(),
			computeToken: ( tokenId, position, computeLogits ) => this.computeToken( renderer, tokenId, position, computeLogits ),
			readLogits: () => this.readLogits( renderer )
		} );

	}

}

export { QwenTSLRunner };

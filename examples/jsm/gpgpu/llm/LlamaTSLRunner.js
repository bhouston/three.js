import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { sampleTopK } from './LLMMath.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { TSLGatedMLP } from './TSLGatedMLP.js';
import { TSLLinear } from './TSLLinear.js';
import { TSLRMSNorm } from './TSLRMSNorm.js';
import { LlamaWeights } from './LlamaWeights.js';

/**
 * Llama-style text generation runner backed by TSL compute kernels.
 *
 * @three_import import { LlamaTSLRunner } from 'three/addons/gpgpu/llm/LlamaTSLRunner.js';
 */
class LlamaTSLRunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( 'LlamaEmbedding' );
		this.layers = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const name = `LlamaLayer${ i }`;
			const ln1 = new TSLRMSNorm( currentNode, block.ln1Weight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: weights.offsetRMSNorm,
				name: `${ name }LN1`,
				workgroupSize: this.workgroupSize
			} );
			const qkv = new TSLLinear( ln1.outputNode, block.attnQKVWeight, block.attnQKVBias, this.hiddenSize, weights.qSize + 2 * weights.kvSize, {
				name: `${ name }QKV`,
				workgroupSize: this.workgroupSize
			} );
			const attention = new TSLAttention( qkv.outputNode, this.hiddenSize, weights.headCount, this.maxTokens, {
				name: `${ name }Attention`,
				workgroupSize: this.workgroupSize,
				headDim: weights.headDim,
				kvHeadCount: weights.kvHeadCount,
				ropeTheta: weights.ropeTheta,
				rotaryDim: weights.headDim
			} );
			const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, block.attnProjBias, weights.qSize, this.hiddenSize, {
				name: `${ name }AttnProj`,
				workgroupSize: this.workgroupSize
			} );
			const addAttention = new TSLAdd( currentNode, attnProj.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const ln2 = new TSLRMSNorm( addAttention.outputNode, block.ln2Weight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: weights.offsetRMSNorm,
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

			this.layers.push( { ln1, qkv, attention, attnProj, addAttention, ln2, mlp, addMLP } );
			currentNode = addMLP.outputNode;

		}

		this.finalNorm = new TSLRMSNorm( currentNode, weights.tensor( 'norm.weight' ), this.hiddenSize, {
			epsilon: weights.rmsNormEps,
			offsetWeight: weights.offsetRMSNorm,
			name: 'LlamaFinalNorm',
			workgroupSize: this.workgroupSize
		} );
		this.logits = this.createLogitLayers();

	}

	static async fromURL( baseURL, options ) {

		return new LlamaTSLRunner( await LlamaWeights.fromURL( baseURL ), options );

	}

	computeToken( renderer, tokenId, position ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;

		for ( const layer of this.layers ) {

			layer.ln1.compute( renderer );
			layer.qkv.compute( renderer );
			layer.attention.compute( renderer, position );
			layer.attnProj.compute( renderer );
			layer.addAttention.compute( renderer );
			layer.ln2.compute( renderer );
			layer.mlp.compute( renderer );
			layer.addMLP.compute( renderer );

		}

		this.finalNorm.compute( renderer );

		for ( const logit of this.logits ) {

			logit.layer.compute( renderer );

		}

	}

	async readLogits( renderer ) {

		const logits = new Float32Array( this.weights.vocabSize );

		for ( const logit of this.logits ) {

			const chunk = new Float32Array( await renderer.getArrayBufferAsync( logit.layer.outputAttribute ) );
			logits.set( chunk.subarray( 0, logit.size ), logit.offset );

		}

		return logits;

	}

	createLogitLayers() {

		const logits = [];
		const { weights, hiddenSize, logitChunkSize } = this;

		for ( let offset = 0; offset < weights.vocabSize; offset += logitChunkSize ) {

			const size = Math.min( logitChunkSize, weights.vocabSize - offset );
			const chunkWeight = new Float32Array( hiddenSize * size );

			for ( let i = 0; i < hiddenSize; i ++ ) {

				const sourceOffset = i * weights.vocabSize + offset;
				chunkWeight.set( weights.logitWeight.subarray( sourceOffset, sourceOffset + size ), i * size );

			}

			logits.push( {
				offset,
				size,
				layer: new TSLLinear( this.finalNorm.outputNode, chunkWeight, null, hiddenSize, size, {
					name: `LlamaLogits${ offset }`,
					workgroupSize: 256
				} )
			} );

		}

		return logits;

	}

	async generate( renderer, prompt, options = {} ) {

		const { inputTokens, newTokenBudget } = this.weights.prepareGeneration(
			prompt,
			this.maxTokens,
			options.maxNewTokens || 32
		);
		const allTokens = inputTokens.slice();
		const generatedTokens = [];
		const signal = options.signal;
		let logits = null;

		for ( let i = 0; i < inputTokens.length; i ++ ) {

			if ( signal !== undefined && signal.aborted ) break;

			this.computeToken( renderer, inputTokens[ i ], i );
			logits = await this.readLogits( renderer );

		}

		for ( let i = 0; i < newTokenBudget; i ++ ) {

			if ( signal !== undefined && signal.aborted ) break;

			const nextToken = sampleTopK( logits, { ...options, tokens: allTokens } );

			if ( nextToken === this.weights.endOfTextTokenId ) break;

			allTokens.push( nextToken );
			generatedTokens.push( nextToken );

			if ( options.onToken ) {

				options.onToken( this.weights.tokenizer.decode( allTokens ), nextToken );

			}

			this.computeToken( renderer, nextToken, allTokens.length - 1 );
			logits = await this.readLogits( renderer );

		}

		return {
			tokens: allTokens,
			generatedTokens,
			text: this.weights.tokenizer.decode( allTokens ),
			generatedText: this.weights.tokenizer.decode( generatedTokens ),
			aborted: signal !== undefined && signal.aborted
		};

	}

}

export { LlamaTSLRunner };

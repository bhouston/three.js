import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { sampleTopK } from './LLMMath.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { TSLGatedMLP } from './TSLGatedMLP.js';
import { TSLLinear } from './TSLLinear.js';
import { TSLRMSNorm } from './TSLRMSNorm.js';
import { GemmaWeights } from './GemmaWeights.js';

/**
 * Gemma 3 text generation runner backed by TSL compute kernels.
 *
 * @three_import import { GemmaTSLRunner } from 'three/addons/gpgpu/llm/GemmaTSLRunner.js';
 */
class GemmaTSLRunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( 'GemmaEmbedding' );
		this.layers = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const name = `GemmaLayer${ i }`;
			const ln1 = new TSLRMSNorm( currentNode, block.ln1Weight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }LN1`,
				workgroupSize: this.workgroupSize
			} );
			const qkv = new TSLLinear( ln1.outputNode, block.attnQKVWeight, null, this.hiddenSize, weights.qSize + 2 * weights.kvSize, {
				name: `${ name }QKV`,
				workgroupSize: this.workgroupSize
			} );
			const attention = new TSLAttention( qkv.outputNode, this.hiddenSize, weights.headCount, this.maxTokens, {
				name: `${ name }Attention`,
				workgroupSize: this.workgroupSize,
				headDim: weights.headDim,
				kvHeadCount: weights.kvHeadCount,
				ropeTheta: block.ropeTheta,
				rotaryDim: weights.headDim,
				slidingWindow: block.slidingWindow,
				attnScale: weights.attnScale,
				qNormWeight: block.qNormWeight,
				kNormWeight: block.kNormWeight,
				rmsEpsilon: weights.rmsNormEps,
				offsetRMSNorm: true
			} );
			const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, null, weights.qSize, this.hiddenSize, {
				name: `${ name }AttnProj`,
				workgroupSize: this.workgroupSize
			} );
			const postAttnNorm = new TSLRMSNorm( attnProj.outputNode, block.postAttnNormWeight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }PostAttn`,
				workgroupSize: this.workgroupSize
			} );
			const addAttention = new TSLAdd( currentNode, postAttnNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const preMlp = new TSLRMSNorm( addAttention.outputNode, block.preMlpNormWeight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }PreMLP`,
				workgroupSize: this.workgroupSize
			} );
			const mlp = new TSLGatedMLP( preMlp.outputNode, block.mlpGateWeight, block.mlpUpWeight, block.mlpDownWeight, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: weights.mlpActivation
			} );
			const postMlpNorm = new TSLRMSNorm( mlp.outputNode, block.postMlpNormWeight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: true,
				name: `${ name }PostMLP`,
				workgroupSize: this.workgroupSize
			} );
			const addMLP = new TSLAdd( addAttention.outputNode, postMlpNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			this.layers.push( { ln1, qkv, attention, attnProj, postAttnNorm, addAttention, preMlp, mlp, postMlpNorm, addMLP } );
			currentNode = addMLP.outputNode;

		}

		this.finalNorm = new TSLRMSNorm( currentNode, weights.tensor( 'norm.weight' ), this.hiddenSize, {
			epsilon: weights.rmsNormEps,
			offsetWeight: true,
			name: 'GemmaFinalNorm',
			workgroupSize: this.workgroupSize
		} );
		this.logits = this.createLogitLayers();

	}

	static async fromURL( baseURL, options ) {

		return new GemmaTSLRunner( await GemmaWeights.fromURL( baseURL ), options );

	}

	computeToken( renderer, tokenId, position ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;

		for ( const layer of this.layers ) {

			layer.ln1.compute( renderer );
			layer.qkv.compute( renderer );
			layer.attention.compute( renderer, position );
			layer.attnProj.compute( renderer );
			layer.postAttnNorm.compute( renderer );
			layer.addAttention.compute( renderer );
			layer.preMlp.compute( renderer );
			layer.mlp.compute( renderer );
			layer.postMlpNorm.compute( renderer );
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
					name: `GemmaLogits${ offset }`,
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

export { GemmaTSLRunner };

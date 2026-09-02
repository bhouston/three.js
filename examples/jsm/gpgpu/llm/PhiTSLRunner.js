import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { sampleTopK } from './LLMMath.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { TSLLinear } from './TSLLinear.js';
import { TSLMLP } from './TSLMLP.js';
import { TSLNormalize } from './TSLNormalize.js';
import { PhiWeights } from './PhiWeights.js';

/**
 * Phi-1/Phi-2 text generation runner backed by TSL compute kernels.
 *
 * @three_import import { PhiTSLRunner } from 'three/addons/gpgpu/llm/PhiTSLRunner.js';
 */
class PhiTSLRunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( 'PhiEmbedding' );
		this.layers = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const name = `PhiLayer${ i }`;
			const ln = new TSLNormalize( currentNode, block.lnWeight, block.lnBias, this.hiddenSize, {
				epsilon: weights.layerNormEps,
				name: `${ name }LN`,
				workgroupSize: this.workgroupSize
			} );
			const qkv = new TSLLinear( ln.outputNode, block.attnQKVWeight, block.attnQKVBias, this.hiddenSize, weights.qSize + 2 * weights.kvSize, {
				name: `${ name }QKV`,
				workgroupSize: this.workgroupSize
			} );
			const attention = new TSLAttention( qkv.outputNode, this.hiddenSize, weights.headCount, this.maxTokens, {
				name: `${ name }Attention`,
				workgroupSize: this.workgroupSize,
				headDim: weights.headDim,
				kvHeadCount: weights.kvHeadCount,
				ropeTheta: weights.ropeTheta,
				rotaryDim: weights.rotaryDim
			} );
			const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, block.attnProjBias, weights.qSize, this.hiddenSize, {
				name: `${ name }AttnProj`,
				workgroupSize: this.workgroupSize
			} );
			const mlp = new TSLMLP( ln.outputNode, block.mlpFCWeight, block.mlpFCBias, block.mlpProjWeight, block.mlpProjBias, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize
			} );
			const addAttention = new TSLAdd( currentNode, attnProj.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const addMLP = new TSLAdd( addAttention.outputNode, mlp.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			this.layers.push( { ln, qkv, attention, attnProj, mlp, addAttention, addMLP } );
			currentNode = addMLP.outputNode;

		}

		this.finalNorm = new TSLNormalize( currentNode, weights.tensor( 'final_layernorm.weight' ), weights.tensor( 'final_layernorm.bias' ), this.hiddenSize, {
			epsilon: weights.layerNormEps,
			name: 'PhiFinalNorm',
			workgroupSize: this.workgroupSize
		} );
		this.logits = this.createLogitLayers();

	}

	static async fromURL( baseURL, options ) {

		return new PhiTSLRunner( await PhiWeights.fromURL( baseURL ), options );

	}

	computeToken( renderer, tokenId, position ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;

		for ( const layer of this.layers ) {

			layer.ln.compute( renderer );
			layer.qkv.compute( renderer );
			layer.attention.compute( renderer, position );
			layer.attnProj.compute( renderer );
			layer.mlp.compute( renderer );
			layer.addAttention.compute( renderer );
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
					name: `PhiLogits${ offset }`,
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
		let logits = null;

		for ( let i = 0; i < inputTokens.length; i ++ ) {

			this.computeToken( renderer, inputTokens[ i ], i );
			logits = await this.readLogits( renderer );

		}

		for ( let i = 0; i < newTokenBudget; i ++ ) {

			const nextToken = sampleTopK( logits, options );

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
			generatedText: this.weights.tokenizer.decode( generatedTokens )
		};

	}

}

export { PhiTSLRunner };

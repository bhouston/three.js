import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { sampleTopK } from './LLMMath.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { TSLConcat } from './TSLConcat.js';
import { TSLGatedDeltaNet } from './TSLGatedDeltaNet.js';
import { TSLGatedMLP } from './TSLGatedMLP.js';
import { TSLLinear } from './TSLLinear.js';
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
				mixer = { qGate, split, kv, packed, attention, attnProj, outputNode: attnProj.outputNode, compute: ( renderer, position ) => {

					qGate.compute( renderer );
					split.compute( renderer );
					kv.compute( renderer );
					packed.compute( renderer );
					attention.compute( renderer, position );
					attnProj.compute( renderer );

				} };

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

		this.finalNorm = new TSLRMSNorm( currentNode, weights.tensor( 'norm.weight' ), this.hiddenSize, {
			epsilon: weights.rmsNormEps,
			offsetWeight: true,
			name: 'QwenFinalNorm',
			workgroupSize: this.workgroupSize
		} );
		this.logits = this.createLogitLayers();

	}

	static async fromURL( baseURL, options ) {

		return new QwenTSLRunner( await QwenWeights.fromURL( baseURL ), options );

	}

	computeToken( renderer, tokenId, position ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;

		for ( const layer of this.layers ) {

			layer.ln1.compute( renderer );

			if ( layer.layerType === 'linear_attention' ) layer.mixer.compute( renderer );
			else layer.mixer.compute( renderer, position );

			layer.addAttention.compute( renderer );
			layer.ln2.compute( renderer );
			layer.mlp.compute( renderer );
			layer.addMLP.compute( renderer );

		}

		this.finalNorm.compute( renderer );

		for ( const logit of this.logits ) logit.layer.compute( renderer );

	}

	async readLogits( renderer ) {

		const logits = new Float32Array( this.weights.vocabSize );

		for ( const logit of this.logits ) {

			const chunk = new Float32Array( await renderer.getArrayBufferAsync( logit.layer.outputAttribute ) );
			logits.set( chunk.subarray( 0, logit.size ), logit.offset );

		}

		return logits;

	}

	resetCaches() {

		for ( const layer of this.layers ) {

			if ( layer.layerType === 'linear_attention' ) layer.mixer.reset();
			else layer.mixer.attention.reset();

		}

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
					name: `QwenLogits${ offset }`,
					workgroupSize: 256
				} )
			} );

		}

		return logits;

	}

	async generate( renderer, prompt, options = {} ) {

		this.resetCaches();

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
			if ( options.onToken ) options.onToken( this.weights.tokenizer.decode( allTokens ), nextToken );
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

export { QwenTSLRunner };

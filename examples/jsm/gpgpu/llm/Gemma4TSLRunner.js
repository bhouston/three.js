import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';

import { linear, logitSoftcap, rmsNorm, sampleTopK } from './LLMMath.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { TSLGatedMLP } from './TSLGatedMLP.js';
import { TSLGELU } from './TSLGELU.js';
import { TSLLinear } from './TSLLinear.js';
import { TSLMul } from './TSLMul.js';
import { TSLRMSNorm } from './TSLRMSNorm.js';
import { Gemma4Weights } from './Gemma4Weights.js';

/**
 * Gemma 4 text generation runner backed by TSL compute kernels.
 *
 * @three_import import { Gemma4TSLRunner } from 'three/addons/gpgpu/llm/Gemma4TSLRunner.js';
 */
class Gemma4TSLRunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( 'Gemma4Embedding' );
		this.pleCombined = weights.pleDim > 0 ? new Float32Array( weights.layerCount * weights.pleDim ) : null;
		this.layers = [];
		this.attentionByLayer = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const name = `Gemma4Layer${ i }`;
			const ln1 = new TSLRMSNorm( currentNode, block.ln1Weight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: false,
				name: `${ name }LN1`,
				workgroupSize: this.workgroupSize
			} );

			let qkv;
			let attention;

			if ( block.isKVShared ) {

				qkv = new TSLLinear( ln1.outputNode, block.qWeight, null, this.hiddenSize, block.qSize, {
					name: `${ name }Q`,
					workgroupSize: this.workgroupSize
				} );
				attention = new TSLAttention( qkv.outputNode, block.qSize, weights.headCount, this.maxTokens, {
					name: `${ name }Attention`,
					workgroupSize: this.workgroupSize,
					headDim: block.headDim,
					kvHeadCount: weights.kvHeadCount,
					ropeTheta: block.ropeTheta,
					rotaryDim: block.rotaryDim,
					ropeFreqDim: block.ropeFreqDim,
					ropePairCount: block.ropePairCount,
					slidingWindow: block.slidingWindow,
					attnScale: weights.attnScale,
					qNormWeight: block.qNormWeight,
					rmsEpsilon: weights.rmsNormEps,
					offsetRMSNorm: false,
					sharedAttention: this.attentionByLayer[ block.sharedSource ]
				} );

			} else {

				qkv = new TSLLinear( ln1.outputNode, block.attnQKVWeight, null, this.hiddenSize, block.qSize + 2 * block.kvSize, {
					name: `${ name }QKV`,
					workgroupSize: this.workgroupSize
				} );
				attention = new TSLAttention( qkv.outputNode, block.qSize, weights.headCount, this.maxTokens, {
					name: `${ name }Attention`,
					workgroupSize: this.workgroupSize,
					headDim: block.headDim,
					kvHeadCount: weights.kvHeadCount,
					ropeTheta: block.ropeTheta,
					rotaryDim: block.rotaryDim,
					ropeFreqDim: block.ropeFreqDim,
					ropePairCount: block.ropePairCount,
					slidingWindow: block.slidingWindow,
					attnScale: weights.attnScale,
					qNormWeight: block.qNormWeight,
					kNormWeight: block.kNormWeight,
					rmsEpsilon: weights.rmsNormEps,
					offsetRMSNorm: false,
					vNorm: true
				} );

			}

			this.attentionByLayer[ i ] = attention;

			const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, null, block.qSize, this.hiddenSize, {
				name: `${ name }AttnProj`,
				workgroupSize: this.workgroupSize
			} );
			const postAttnNorm = new TSLRMSNorm( attnProj.outputNode, block.postAttnNormWeight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: false,
				name: `${ name }PostAttn`,
				workgroupSize: this.workgroupSize
			} );
			const addAttention = new TSLAdd( currentNode, postAttnNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const preMlp = new TSLRMSNorm( addAttention.outputNode, block.preMlpNormWeight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: false,
				name: `${ name }PreMLP`,
				workgroupSize: this.workgroupSize
			} );
			const mlp = new TSLGatedMLP( preMlp.outputNode, block.mlpGateWeight, block.mlpUpWeight, block.mlpDownWeight, this.hiddenSize, block.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: weights.mlpActivation
			} );
			const postMlpNorm = new TSLRMSNorm( mlp.outputNode, block.postMlpNormWeight, this.hiddenSize, {
				epsilon: weights.rmsNormEps,
				offsetWeight: false,
				name: `${ name }PostMLP`,
				workgroupSize: this.workgroupSize
			} );
			const addMLP = new TSLAdd( addAttention.outputNode, postMlpNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			let stream = addMLP.outputNode;
			let ple = null;

			if ( weights.pleDim > 0 ) {

				const pleInputBuffer = new Float32Array( weights.pleDim );
				const pleInputAttribute = new StorageBufferAttribute( pleInputBuffer, 1 );
				const pleInputNode = storage( pleInputAttribute, 'float', weights.pleDim ).setName( `${ name }PLEIn` );
				const pleGate = new TSLLinear( addMLP.outputNode, block.pleGateWeight, null, this.hiddenSize, weights.pleDim, {
					name: `${ name }PLEGate`,
					workgroupSize: this.workgroupSize
				} );
				const pleGelu = new TSLGELU( pleGate.outputNode, weights.pleDim, {
					name: `${ name }PLEGELU`,
					workgroupSize: this.workgroupSize
				} );
				const pleMul = new TSLMul( pleGelu.outputNode, pleInputNode, weights.pleDim, {
					name: `${ name }PLEMul`
				} );
				const pleProj = new TSLLinear( pleMul.outputNode, block.pleProjWeight, null, weights.pleDim, this.hiddenSize, {
					name: `${ name }PLEProj`,
					workgroupSize: this.workgroupSize
				} );
				const pleNorm = new TSLRMSNorm( pleProj.outputNode, block.pleNormWeight, this.hiddenSize, {
					epsilon: weights.rmsNormEps,
					offsetWeight: false,
					name: `${ name }PLENorm`,
					workgroupSize: this.workgroupSize
				} );
				const addPLE = new TSLAdd( addMLP.outputNode, pleNorm.outputNode, this.hiddenSize, {
					name: `${ name }AddPLE`,
					workgroupSize: this.workgroupSize
				} );
				ple = { pleInputBuffer, pleInputAttribute, pleGate, pleGelu, pleMul, pleProj, pleNorm, addPLE };
				stream = addPLE.outputNode;

			}

			this.layers.push( { ln1, qkv, attention, attnProj, postAttnNorm, addAttention, preMlp, mlp, postMlpNorm, addMLP, ple, isKVShared: block.isKVShared } );
			currentNode = stream;

		}

		this.finalNorm = new TSLRMSNorm( currentNode, weights.tensor( 'norm.weight' ), this.hiddenSize, {
			epsilon: weights.rmsNormEps,
			offsetWeight: false,
			name: 'Gemma4FinalNorm',
			workgroupSize: this.workgroupSize
		} );
		this.logits = this.createLogitLayers();

	}

	static async fromURL( baseURL, options ) {

		return new Gemma4TSLRunner( await Gemma4Weights.fromURL( baseURL ), options );

	}

	fillPerLayerInputs( tokenId ) {

		const { weights, embeddingBuffer, pleCombined } = this;
		if ( pleCombined === null ) return;

		const lookup = weights.perLayerTokenEmbedding( tokenId );
		const projected = linear( embeddingBuffer, weights.perLayerProjectionWeight, null, weights.hiddenSize, weights.layerCount * weights.pleDim );

		for ( let i = 0; i < projected.length; i ++ ) projected[ i ] *= weights.perLayerModelProjectionScale;

		for ( let layer = 0; layer < weights.layerCount; layer ++ ) {

			const offset = layer * weights.pleDim;
			const slice = projected.subarray( offset, offset + weights.pleDim );
			const normed = rmsNorm( slice, weights.perLayerProjectionNorm, weights.rmsNormEps, false );

			for ( let i = 0; i < weights.pleDim; i ++ ) {

				pleCombined[ offset + i ] = ( normed[ i ] + lookup[ offset + i ] ) * weights.perLayerInputScale;

			}

		}

	}

	computeToken( renderer, tokenId, position ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;
		this.fillPerLayerInputs( tokenId );

		for ( let i = 0; i < this.layers.length; i ++ ) {

			const layer = this.layers[ i ];
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

			if ( layer.ple ) {

				layer.ple.pleInputBuffer.set( this.pleCombined.subarray( i * this.weights.pleDim, ( i + 1 ) * this.weights.pleDim ) );
				layer.ple.pleInputAttribute.needsUpdate = true;
				layer.ple.pleGate.compute( renderer );
				layer.ple.pleGelu.compute( renderer );
				layer.ple.pleMul.compute( renderer );
				layer.ple.pleProj.compute( renderer );
				layer.ple.pleNorm.compute( renderer );
				layer.ple.addPLE.compute( renderer );

			}

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

		return logitSoftcap( logits, this.weights.finalLogitSoftcapping );

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
					name: `Gemma4Logits${ offset }`,
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

			const nextToken = sampleTopK( logits, options );
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

export { Gemma4TSLRunner };

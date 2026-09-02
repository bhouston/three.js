import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, instanceIndex, storage, uint } from 'three/tsl';

import { DecoderWeights } from './DecoderWeights.js';
import { generateAsync } from './LLMGenerate.js';
import { logitSoftcap } from './LLMMath.js';
import { TSLAdd } from './TSLAdd.js';
import { TSLAttention } from './TSLAttention.js';
import { orderedComputeNodes } from './TSLCompute.js';
import { TSLGatedMLP } from './TSLGatedMLP.js';
import { TSLLinear } from './TSLLinear.js';
import { createChunkedLogitLayers, createLogitSampler, readChunkedLogits } from './TSLLogits.js';
import { TSLMLP } from './TSLMLP.js';
import { TSLNormalize } from './TSLNormalize.js';
import { TSLRMSNorm } from './TSLRMSNorm.js';

/**
 * Parameterized TSL decoder for GPT-2, Llama-family, Phi, and Gemma 3.
 *
 * @three_import import { DecoderTSLRunner } from 'three/addons/gpgpu/llm/DecoderTSLRunner.js';
 */
class DecoderTSLRunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.recipe = weights.recipe;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.workgroupSize = options.workgroupSize || 64;
		this.logitChunkSize = options.logitChunkSize || 8192;
		this.prefillChunkSize = options.prefillChunkSize || 32;
		this.hiddenSize = weights.hiddenSize;
		this.embeddingBuffer = new Float32Array( this.hiddenSize );
		this.embeddingAttribute = new StorageBufferAttribute( this.embeddingBuffer, 1 );
		this.embeddingNode = storage( this.embeddingAttribute, 'float', this.hiddenSize ).setName( `${ weights.architecture }Embedding` );
		this.positionBuffer = new Uint32Array( 1 );
		this.positionAttribute = new StorageBufferAttribute( this.positionBuffer, 1 );
		this.positionNode = storage( this.positionAttribute, 'uint', 1 ).setName( `${ weights.architecture }Position` );
		this.prefillCursorBuffer = new Uint32Array( 1 );
		this.prefillCursorAttribute = new StorageBufferAttribute( this.prefillCursorBuffer, 1 );
		this.prefillCursorNode = storage( this.prefillCursorAttribute, 'uint', 1 ).setName( `${ weights.architecture }PrefillCursor` );
		this.prefillEmbeddingBuffer = new Float32Array( this.prefillChunkSize * this.hiddenSize );
		this.prefillEmbeddingAttribute = new StorageBufferAttribute( this.prefillEmbeddingBuffer, 1 );
		this.prefillEmbeddingNode = storage( this.prefillEmbeddingAttribute, 'float', this.prefillEmbeddingBuffer.length ).setName( `${ weights.architecture }PrefillEmbeddings` );
		this.prefillCopyNode = this.createPrefillCopyNode( `${ weights.architecture }PrefillCopy` );
		this.prefillAdvanceNode = this.createPrefillAdvanceNode( `${ weights.architecture }PrefillAdvance` );
		this.layers = [];

		let currentNode = this.embeddingNode;

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const built = this.buildLayer( weights.block( i ), i, currentNode );
			this.layers.push( built );
			currentNode = built.outputNode;

		}

		this.finalNorm = this.buildFinalNorm( currentNode );
		this.logits = createChunkedLogitLayers( this.finalNorm.outputNode, weights, this.logitChunkSize, `${ weights.architecture }Logits` );
		this.logitSampler = createLogitSampler( this.logits, {
			candidateCount: options.logitCandidateCount || 8,
			logitSoftcap: this.recipe.finalLogitSoftcap,
			name: `${ weights.architecture }Logits`
		} );
		this.prefillComputeNodes = this.createComputeNodes( false );
		this.computeNodes = this.createComputeNodes();

	}

	static async fromURL( baseURL, options ) {

		return new this( await DecoderWeights.fromURL( baseURL, options ), options );

	}

	buildNorm( node, weight, bias, name ) {

		if ( this.recipe.norm === 'layer_norm' ) {

			return new TSLNormalize( node, weight, bias, this.hiddenSize, {
				epsilon: this.recipe.normEps,
				name,
				workgroupSize: this.workgroupSize
			} );

		}

		return new TSLRMSNorm( node, weight, this.hiddenSize, {
			epsilon: this.recipe.normEps,
			offsetWeight: this.recipe.norm === 'rms_offset',
			name,
			workgroupSize: this.workgroupSize
		} );

	}

	buildAttention( qkvNode, block, name ) {

		const { weights, recipe } = this;

		return new TSLAttention( qkvNode, this.hiddenSize, weights.headCount, this.maxTokens, {
			name: `${ name }Attention`,
			workgroupSize: this.workgroupSize,
			headDim: weights.headDim,
			kvHeadCount: weights.kvHeadCount,
			ropeTheta: block.ropeTheta !== undefined ? block.ropeTheta : recipe.ropeTheta,
			rotaryDim: recipe.rotaryDim || weights.headDim,
			slidingWindow: block.slidingWindow || 0,
			attnScale: recipe.attnScale,
			qNormWeight: block.qNormWeight,
			kNormWeight: block.kNormWeight,
			rmsEpsilon: recipe.normEps,
			offsetRMSNorm: recipe.norm === 'rms_offset',
			positionNode: this.positionNode.element( uint( 0 ) )
		} );

	}

	createPrefillCopyNode( name ) {

		const { hiddenSize, workgroupSize, embeddingNode, prefillCursorNode, prefillEmbeddingNode } = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );

			If( dim.lessThan( uint( hiddenSize ) ), () => {

				const offset = prefillCursorNode.element( uint( 0 ) ).mul( uint( hiddenSize ) ).add( dim );
				embeddingNode.element( dim ).assign( prefillEmbeddingNode.element( offset ) );

			} );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	createPrefillAdvanceNode( name ) {

		const { prefillCursorNode, positionNode } = this;

		return Fn( () => {

			prefillCursorNode.element( uint( 0 ) ).assign( prefillCursorNode.element( uint( 0 ) ).add( uint( 1 ) ) );
			positionNode.element( uint( 0 ) ).assign( positionNode.element( uint( 0 ) ).add( uint( 1 ) ) );

		} )().compute( 1, [ 1 ] ).setName( name );

	}

	buildLayer( block, index, residualNode ) {

		const { weights, recipe } = this;
		const name = `${ weights.architecture }Layer${ index }`;

		if ( recipe.residual === 'parallel' ) {

			const ln = this.buildNorm( residualNode, block.lnWeight, block.lnBias, `${ name }LN` );
			const qkv = new TSLLinear( ln.outputNode, block.attnQKVWeight, block.attnQKVBias, this.hiddenSize, weights.qSize + 2 * weights.kvSize, {
				name: `${ name }QKV`,
				workgroupSize: this.workgroupSize
			} );
			const attention = this.buildAttention( qkv.outputNode, block, name );
			const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, block.attnProjBias, weights.qSize, this.hiddenSize, {
				name: `${ name }AttnProj`,
				workgroupSize: this.workgroupSize
			} );
			const mlp = new TSLMLP( ln.outputNode, block.mlpFCWeight, block.mlpFCBias, block.mlpProjWeight, block.mlpProjBias, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize
			} );
			const addAttention = new TSLAdd( residualNode, attnProj.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const addMLP = new TSLAdd( addAttention.outputNode, mlp.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			return {
				kind: 'parallel',
				ln, qkv, attention, attnProj, mlp, addAttention, addMLP,
				outputNode: addMLP.outputNode
			};

		}

		const ln1 = this.buildNorm( residualNode, block.ln1Weight, block.ln1Bias || null, `${ name }LN1` );
		const qkvOut = recipe.architecture === 'gpt2' ? this.hiddenSize * 3 : weights.qSize + 2 * weights.kvSize;
		const qkv = new TSLLinear( ln1.outputNode, block.attnQKVWeight, block.attnQKVBias || null, this.hiddenSize, qkvOut, {
			name: `${ name }QKV`,
			workgroupSize: this.workgroupSize
		} );
		const attention = this.buildAttention( qkv.outputNode, block, name );
		const attnIn = recipe.architecture === 'gpt2' ? this.hiddenSize : weights.qSize;
		const attnProj = new TSLLinear( attention.outputNode, block.attnProjWeight, block.attnProjBias || null, attnIn, this.hiddenSize, {
			name: `${ name }AttnProj`,
			workgroupSize: this.workgroupSize
		} );

		if ( recipe.postNorms ) {

			const postAttnNorm = this.buildNorm( attnProj.outputNode, block.postAttnNormWeight, null, `${ name }PostAttn` );
			const addAttention = new TSLAdd( residualNode, postAttnNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddAttention`,
				workgroupSize: this.workgroupSize
			} );
			const preMlp = this.buildNorm( addAttention.outputNode, block.preMlpNormWeight, null, `${ name }PreMLP` );
			const mlp = new TSLGatedMLP( preMlp.outputNode, block.mlpGateWeight, block.mlpUpWeight, block.mlpDownWeight, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: recipe.mlpActivation
			} );
			const postMlpNorm = this.buildNorm( mlp.outputNode, block.postMlpNormWeight, null, `${ name }PostMLP` );
			const addMLP = new TSLAdd( addAttention.outputNode, postMlpNorm.outputNode, this.hiddenSize, {
				name: `${ name }AddMLP`,
				workgroupSize: this.workgroupSize
			} );

			return {
				kind: 'gemma',
				ln1, qkv, attention, attnProj, postAttnNorm, addAttention, preMlp, mlp, postMlpNorm, addMLP,
				outputNode: addMLP.outputNode
			};

		}

		const addAttention = new TSLAdd( residualNode, attnProj.outputNode, this.hiddenSize, {
			name: `${ name }AddAttention`,
			workgroupSize: this.workgroupSize
		} );
		const ln2 = this.buildNorm( addAttention.outputNode, block.ln2Weight, block.ln2Bias || null, `${ name }LN2` );
		const mlp = recipe.mlp === 'dense_gelu'
			? new TSLMLP( ln2.outputNode, block.mlpFCWeight, block.mlpFCBias, block.mlpProjWeight, block.mlpProjBias, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize
			} )
			: new TSLGatedMLP( ln2.outputNode, block.mlpGateWeight, block.mlpUpWeight, block.mlpDownWeight, this.hiddenSize, weights.innerSize, {
				name: `${ name }MLP`,
				workgroupSize: this.workgroupSize,
				activation: recipe.mlpActivation
			} );
		const addMLP = new TSLAdd( addAttention.outputNode, mlp.outputNode, this.hiddenSize, {
			name: `${ name }AddMLP`,
			workgroupSize: this.workgroupSize
		} );

		return {
			kind: 'sequential',
			ln1, qkv, attention, attnProj, addAttention, ln2, mlp, addMLP,
			outputNode: addMLP.outputNode
		};

	}

	buildFinalNorm( node ) {

		return this.buildNorm( node, this.weights.outputNormWeight, this.weights.outputNormBias, `${ this.weights.architecture }FinalNorm` );

	}

	createComputeNodes( includeLogits = true ) {

		const nodes = [];

		for ( const layer of this.layers ) {

			if ( layer.kind === 'parallel' ) {

				nodes.push( ...orderedComputeNodes(
					layer.ln, layer.qkv, layer.attention, layer.attnProj, layer.mlp, layer.addAttention, layer.addMLP
				) );

			} else if ( layer.kind === 'gemma' ) {

				nodes.push( ...orderedComputeNodes(
					layer.ln1, layer.qkv, layer.attention, layer.attnProj, layer.postAttnNorm,
					layer.addAttention, layer.preMlp, layer.mlp, layer.postMlpNorm, layer.addMLP
				) );

			} else {

				nodes.push( ...orderedComputeNodes(
					layer.ln1, layer.qkv, layer.attention, layer.attnProj,
					layer.addAttention, layer.ln2, layer.mlp, layer.addMLP
				) );

			}

		}

		if ( includeLogits ) {

			nodes.push( ...orderedComputeNodes( this.finalNorm, ...this.logits.map( ( logit ) => logit.layer ) ) );

		}

		return nodes;

	}

	sampleComputeNodes( candidateCount ) {

		return this.computeNodes.concat( this.logitSampler.computeNodesFor( candidateCount ) );

	}

	prefillChunkComputeNodes( count ) {

		const nodes = [];

		for ( let i = 0; i < count; i ++ ) {

			nodes.push( this.prefillCopyNode, ...this.prefillComputeNodes, this.prefillAdvanceNode );

		}

		return nodes;

	}

	setPosition( position ) {

		this.positionBuffer[ 0 ] = position;
		this.positionAttribute.needsUpdate = true;

	}

	computeToken( renderer, tokenId, position, computeLogits = true, sampleCandidateCount = 0 ) {

		this.weights.embedding( tokenId, position, this.embeddingBuffer );
		this.embeddingAttribute.needsUpdate = true;
		this.setPosition( position );

		for ( const layer of this.layers ) layer.attention.setPosition( position );

		renderer.compute( computeLogits
			? ( sampleCandidateCount > 0 ? this.sampleComputeNodes( sampleCandidateCount ) : this.computeNodes )
			: this.prefillComputeNodes );

	}

	async prefillTokens( renderer, inputTokens, start, end, onProgress ) {

		for ( let offset = start; offset < end; offset += this.prefillChunkSize ) {

			const count = Math.min( this.prefillChunkSize, end - offset );

			for ( let i = 0; i < count; i ++ ) {

				this.weights.embedding(
					inputTokens[ offset + i ],
					offset + i,
					this.prefillEmbeddingBuffer.subarray( i * this.hiddenSize, ( i + 1 ) * this.hiddenSize )
				);

			}

			this.prefillEmbeddingAttribute.needsUpdate = true;
			this.prefillCursorBuffer[ 0 ] = 0;
			this.prefillCursorAttribute.needsUpdate = true;
			this.setPosition( offset );
			renderer.compute( this.prefillChunkComputeNodes( count ) );
			if ( onProgress ) await onProgress( offset + count );

		}

	}

	async readLogits( renderer ) {

		const logits = await readChunkedLogits( renderer, this.logits, this.weights.vocabSize );
		return logitSoftcap( logits, this.recipe.finalLogitSoftcap );

	}

	async sampleToken( renderer, candidateCount, options ) {

		return this.logitSampler.sampleToken( renderer, candidateCount, options );

	}

	resetCache() {

		this._cacheTokens = [];
		this._cacheLogits = null;

		for ( const layer of this.layers ) layer.attention.reset();

	}

	async generate( renderer, prompt, options = {} ) {

		return generateAsync( this, prompt, options, {
			rewindable: true,
			resetCache: () => this.resetCache(),
			computeToken: ( tokenId, position, computeLogits, sampleCandidateCount ) => this.computeToken( renderer, tokenId, position, computeLogits, sampleCandidateCount ),
			prefillTokens: ( inputTokens, start, end, onProgress ) => this.prefillTokens( renderer, inputTokens, start, end, onProgress ),
			readLogits: () => this.readLogits( renderer ),
			sampleToken: ( candidateCount, sampleOptions ) => this.sampleToken( renderer, candidateCount, sampleOptions ),
			maxGpuCandidateCount: this.logitSampler.candidateCount
		} );

	}

}

export { DecoderTSLRunner };

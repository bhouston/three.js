import { SafeTensorsLoader } from './SafeTensorsLoader.js';
import { fetchJSON, packProjections, prepareGeneration, tensorToFloat32, transpose2D } from './LLMTensors.js';
import { UnigramTokenizer } from './UnigramTokenizer.js';

/**
 * Loads a Hugging Face Gemma 3 text model (`gemma3_text`).
 *
 * Differences from Llama: embeddings are scaled by `sqrt(hidden)`, Q/K are
 * RMSNorm'd per head, local layers use sliding-window attention and a smaller
 * RoPE base, and each residual branch has a post-norm.
 *
 * @three_import import { GemmaWeights } from 'three/addons/gpgpu/llm/GemmaWeights.js';
 */
class GemmaWeights {

	constructor( config, tensors, tokenizer ) {

		this.architecture = 'gemma3';
		this.config = config;
		this.tensors = tensors;
		this.tokenizer = tokenizer;
		this.tensorPrefix = tensors[ 'model.embed_tokens.weight' ] !== undefined ? 'model.' : '';
		this.hiddenSize = config.hidden_size;
		this.innerSize = config.intermediate_size;
		this.layerCount = config.num_hidden_layers;
		this.headCount = config.num_attention_heads;
		this.kvHeadCount = config.num_key_value_heads || config.num_attention_heads;
		this.headDim = config.head_dim || ( config.hidden_size / config.num_attention_heads );
		this.qSize = this.headCount * this.headDim;
		this.kvSize = this.kvHeadCount * this.headDim;
		this.vocabSize = config.vocab_size;
		this.globalRopeTheta = config.rope_theta || 1000000;
		this.localRopeTheta = config.rope_local_base_freq || 10000;
		this.slidingWindow = config.sliding_window || 512;
		this.layerTypes = config.layer_types || defaultLayerTypes( this.layerCount );
		this.rmsNormEps = config.rms_norm_eps || 1e-6;
		this.offsetRMSNorm = true;
		this.mlpActivation = config.hidden_activation || config.hidden_act || 'gelu_pytorch_tanh';
		this.embedScale = Math.sqrt( this.hiddenSize );
		this.attnScale = ( config.query_pre_attn_scalar || this.headDim ) ** - 0.5;
		this.endOfTextTokenId = config.eos_token_id ?? tokenizer.endOfTextTokenId ?? 1;
		this._float32 = new Map();

		const embedding = this.tensor( 'embed_tokens.weight' );
		const lmHead = this.hasTensor( 'lm_head.weight' ) && config.tie_word_embeddings !== true
			? this.tensor( 'lm_head.weight' )
			: embedding;

		this.logitWeight = transpose2D( lmHead, this.vocabSize, this.hiddenSize );
		this._blocks = [];

		for ( let i = 0; i < this.layerCount; i ++ ) {

			this._blocks[ i ] = this.createBlock( i );

		}

	}

	contextLimit() {

		return Math.min( this.config.max_position_embeddings || 2048, 2048 );

	}

	prepareGeneration( prompt, maxTokens, maxNewTokens ) {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	static async fromURL( baseURL ) {

		const root = baseURL.endsWith( '/' ) ? baseURL : `${ baseURL }/`;
		const [ config, safeTensors, tokenizer ] = await Promise.all( [
			fetchJSON( `${ root }config.json`, 'GemmaWeights' ),
			new SafeTensorsLoader().load( `${ root }model.safetensors` ),
			UnigramTokenizer.fromURL( root )
		] );

		if ( config.bos_token_id !== undefined ) tokenizer.bosTokenId = config.bos_token_id;
		if ( config.eos_token_id !== undefined ) {

			tokenizer.eosTokenId = config.eos_token_id;
			tokenizer.endOfTextTokenId = config.eos_token_id;

		}

		return new GemmaWeights( config, safeTensors.tensors, tokenizer );

	}

	hasTensor( name ) {

		return this.tensors[ `${ this.tensorPrefix }${ name }` ] !== undefined || this.tensors[ name ] !== undefined;

	}

	tensor( name ) {

		if ( this._float32.has( name ) ) return this._float32.get( name );

		const tensor = this.tensors[ `${ this.tensorPrefix }${ name }` ] || this.tensors[ name ];

		if ( tensor === undefined ) {

			throw new Error( `GemmaWeights: Missing tensor "${ this.tensorPrefix }${ name }".` );

		}

		const data = tensorToFloat32( tensor );
		this._float32.set( name, data );

		return data;

	}

	linearWeight( name, outFeatures, inFeatures ) {

		return transpose2D( this.tensor( name ), outFeatures, inFeatures );

	}

	block( index ) {

		return this._blocks[ index ];

	}

	createBlock( index ) {

		const prefix = `layers.${ index }`;
		const { hiddenSize, qSize, kvSize, innerSize } = this;
		const layerType = this.layerTypes[ index ] || 'sliding_attention';
		const q = this.linearWeight( `${ prefix }.self_attn.q_proj.weight`, qSize, hiddenSize );
		const k = this.linearWeight( `${ prefix }.self_attn.k_proj.weight`, kvSize, hiddenSize );
		const v = this.linearWeight( `${ prefix }.self_attn.v_proj.weight`, kvSize, hiddenSize );

		return {
			layerType,
			ropeTheta: layerType === 'full_attention' ? this.globalRopeTheta : this.localRopeTheta,
			slidingWindow: layerType === 'sliding_attention' ? this.slidingWindow : 0,
			ln1Weight: this.tensor( `${ prefix }.input_layernorm.weight` ),
			postAttnNormWeight: this.tensor( `${ prefix }.post_attention_layernorm.weight` ),
			preMlpNormWeight: this.tensor( `${ prefix }.pre_feedforward_layernorm.weight` ),
			postMlpNormWeight: this.tensor( `${ prefix }.post_feedforward_layernorm.weight` ),
			qNormWeight: this.tensor( `${ prefix }.self_attn.q_norm.weight` ),
			kNormWeight: this.tensor( `${ prefix }.self_attn.k_norm.weight` ),
			attnQKVWeight: packProjections( [ q, k, v ], hiddenSize ),
			attnProjWeight: this.linearWeight( `${ prefix }.self_attn.o_proj.weight`, hiddenSize, qSize ),
			mlpGateWeight: this.linearWeight( `${ prefix }.mlp.gate_proj.weight`, innerSize, hiddenSize ),
			mlpUpWeight: this.linearWeight( `${ prefix }.mlp.up_proj.weight`, innerSize, hiddenSize ),
			mlpDownWeight: this.linearWeight( `${ prefix }.mlp.down_proj.weight`, hiddenSize, innerSize )
		};

	}

	embedding( tokenId, position, target = new Float32Array( this.hiddenSize ) ) {

		const tokenEmbedding = this.tensor( 'embed_tokens.weight' );
		const tokenOffset = tokenId * this.hiddenSize;

		for ( let i = 0; i < this.hiddenSize; i ++ ) {

			target[ i ] = tokenEmbedding[ tokenOffset + i ] * this.embedScale;

		}

		return target;

	}

}

function defaultLayerTypes( layerCount ) {

	const types = [];

	for ( let i = 0; i < layerCount; i ++ ) {

		types.push( ( i + 1 ) % 6 === 0 ? 'full_attention' : 'sliding_attention' );

	}

	return types;

}

export { GemmaWeights };

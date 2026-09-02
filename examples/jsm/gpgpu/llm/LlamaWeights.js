import { SafeTensorsLoader } from './SafeTensorsLoader.js';
import { GPT2Tokenizer } from './GPT2Tokenizer.js';
import { fetchJSON, packProjections, prepareGeneration, tensorToFloat32, transpose2D } from './LLMTensors.js';

/**
 * Loads a Hugging Face Llama-style causal LM (SmolLM, TinyLlama, Llama).
 *
 * Linear weights are transposed from PyTorch `[out, in]` into the `[in, out]`
 * layout used by the TSL matmul kernels. Q/K/V projections are packed so the
 * existing fused attention kernels can run grouped-query attention.
 *
 * @three_import import { LlamaWeights } from 'three/addons/gpgpu/llm/LlamaWeights.js';
 */
class LlamaWeights {

	constructor( config, tensors, tokenizer ) {

		this.architecture = 'llama';
		this.config = config;
		this.tensors = tensors;
		this.tokenizer = tokenizer;
		this.tensorPrefix = detectPrefix( tensors );
		this.hiddenSize = config.hidden_size;
		this.innerSize = config.intermediate_size;
		this.layerCount = config.num_hidden_layers;
		this.headCount = config.num_attention_heads;
		this.kvHeadCount = config.num_key_value_heads || config.num_attention_heads;
		this.headDim = config.head_dim || ( config.hidden_size / config.num_attention_heads );
		this.qSize = this.headCount * this.headDim;
		this.kvSize = this.kvHeadCount * this.headDim;
		this.vocabSize = config.vocab_size;
		this.ropeTheta = config.rope_theta || 10000;
		this.rmsNormEps = config.rms_norm_eps || 1e-5;
		this.offsetRMSNorm = config.model_type === 'gemma' || config.model_type === 'gemma2' || config.model_type === 'gemma3_text';
		this.mlpActivation = config.hidden_act || 'silu';
		this.endOfTextTokenId = config.eos_token_id ?? tokenizer.endOfTextTokenId ?? 0;
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

		return this.config.max_position_embeddings || 2048;

	}

	prepareGeneration( prompt, maxTokens, maxNewTokens ) {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	static async fromURL( baseURL ) {

		const root = baseURL.endsWith( '/' ) ? baseURL : `${ baseURL }/`;
		const [ config, safeTensors, tokenizer ] = await Promise.all( [
			fetchJSON( `${ root }config.json`, 'LlamaWeights' ),
			new SafeTensorsLoader().load( `${ root }model.safetensors` ),
			GPT2Tokenizer.fromURLs( `${ root }vocab.json`, `${ root }merges.txt` )
		] );

		return new LlamaWeights( config, safeTensors.tensors, tokenizer );

	}

	hasTensor( name ) {

		return this.tensors[ `${ this.tensorPrefix }${ name }` ] !== undefined || this.tensors[ name ] !== undefined;

	}

	tensor( name ) {

		if ( this._float32.has( name ) ) return this._float32.get( name );

		const tensor = this.tensors[ `${ this.tensorPrefix }${ name }` ] || this.tensors[ name ];

		if ( tensor === undefined ) {

			throw new Error( `LlamaWeights: Missing tensor "${ this.tensorPrefix }${ name }".` );

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
		const q = this.linearWeight( `${ prefix }.self_attn.q_proj.weight`, qSize, hiddenSize );
		const k = this.linearWeight( `${ prefix }.self_attn.k_proj.weight`, kvSize, hiddenSize );
		const v = this.linearWeight( `${ prefix }.self_attn.v_proj.weight`, kvSize, hiddenSize );

		return {
			ln1Weight: this.tensor( `${ prefix }.input_layernorm.weight` ),
			ln2Weight: this.tensor( `${ prefix }.post_attention_layernorm.weight` ),
			attnQKVWeight: packProjections( [ q, k, v ], hiddenSize ),
			attnQKVBias: null,
			attnProjWeight: this.linearWeight( `${ prefix }.self_attn.o_proj.weight`, hiddenSize, qSize ),
			attnProjBias: null,
			mlpGateWeight: this.linearWeight( `${ prefix }.mlp.gate_proj.weight`, innerSize, hiddenSize ),
			mlpUpWeight: this.linearWeight( `${ prefix }.mlp.up_proj.weight`, innerSize, hiddenSize ),
			mlpDownWeight: this.linearWeight( `${ prefix }.mlp.down_proj.weight`, hiddenSize, innerSize )
		};

	}

	embedding( tokenId, position, target = new Float32Array( this.hiddenSize ) ) {

		const tokenEmbedding = this.tensor( 'embed_tokens.weight' );
		const tokenOffset = tokenId * this.hiddenSize;
		target.set( tokenEmbedding.subarray( tokenOffset, tokenOffset + this.hiddenSize ) );

		return target;

	}

}

function detectPrefix( tensors ) {

	if ( tensors[ 'model.embed_tokens.weight' ] !== undefined ) return 'model.';
	if ( tensors[ 'embed_tokens.weight' ] !== undefined ) return '';

	return 'model.';

}

export { LlamaWeights };

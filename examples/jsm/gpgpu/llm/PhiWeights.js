import { SafeTensorsLoader } from './SafeTensorsLoader.js';
import { GPT2Tokenizer } from './GPT2Tokenizer.js';
import { fetchJSON, packBiases, packProjections, prepareGeneration, tensorToFloat32, transpose2D } from './LLMTensors.js';

/**
 * Loads a Hugging Face Phi-1/Phi-2 causal LM.
 *
 * Phi keeps GPT-2's LayerNorm + GELU MLP, but drops learned position
 * embeddings for partial RoPE and runs attention and the MLP in parallel.
 *
 * @three_import import { PhiWeights } from 'three/addons/gpgpu/llm/PhiWeights.js';
 */
class PhiWeights {

	constructor( config, tensors, tokenizer ) {

		this.architecture = 'phi';
		this.config = config;
		this.tensors = tensors;
		this.tokenizer = tokenizer;
		this.tensorPrefix = tensors[ 'model.embed_tokens.weight' ] !== undefined ? 'model.' : '';
		this.hiddenSize = config.hidden_size;
		this.innerSize = config.intermediate_size;
		this.layerCount = config.num_hidden_layers;
		this.headCount = config.num_attention_heads;
		this.kvHeadCount = config.num_key_value_heads || config.num_attention_heads;
		this.headDim = config.hidden_size / config.num_attention_heads;
		this.qSize = this.headCount * this.headDim;
		this.kvSize = this.kvHeadCount * this.headDim;
		this.vocabSize = config.vocab_size;
		this.ropeTheta = config.rope_theta || 10000;
		this.rotaryDim = config.rotary_dim || Math.round( ( config.partial_rotary_factor || 0.5 ) * this.headDim );
		this.layerNormEps = config.layer_norm_eps || 1e-5;
		this.endOfTextTokenId = config.eos_token_id ?? tokenizer.endOfTextTokenId ?? 0;
		this._float32 = new Map();

		this.logitWeight = transpose2D( this.tensor( 'lm_head.weight', true ), this.vocabSize, this.hiddenSize );
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
			fetchJSON( `${ root }config.json`, 'PhiWeights' ),
			new SafeTensorsLoader().load( `${ root }model.safetensors` ),
			GPT2Tokenizer.fromURLs( `${ root }vocab.json`, `${ root }merges.txt` )
		] );

		return new PhiWeights( config, safeTensors.tensors, tokenizer );

	}

	tensor( name, unprefixed = false ) {

		const key = unprefixed ? name : `${ this.tensorPrefix }${ name }`;

		if ( this._float32.has( key ) ) return this._float32.get( key );

		const tensor = this.tensors[ key ] || this.tensors[ name ];

		if ( tensor === undefined ) {

			throw new Error( `PhiWeights: Missing tensor "${ key }".` );

		}

		const data = tensorToFloat32( tensor );
		this._float32.set( key, data );

		return data;

	}

	linear( name, outFeatures, inFeatures ) {

		return transpose2D( this.tensor( name ), outFeatures, inFeatures );

	}

	bias( name, size ) {

		if ( this.tensors[ `${ this.tensorPrefix }${ name }` ] === undefined && this.tensors[ name ] === undefined ) {

			return new Float32Array( size );

		}

		return this.tensor( name );

	}

	block( index ) {

		return this._blocks[ index ];

	}

	createBlock( index ) {

		const prefix = `layers.${ index }`;
		const { hiddenSize, qSize, kvSize, innerSize } = this;
		const q = this.linear( `${ prefix }.self_attn.q_proj.weight`, qSize, hiddenSize );
		const k = this.linear( `${ prefix }.self_attn.k_proj.weight`, kvSize, hiddenSize );
		const v = this.linear( `${ prefix }.self_attn.v_proj.weight`, kvSize, hiddenSize );

		return {
			lnWeight: this.tensor( `${ prefix }.input_layernorm.weight` ),
			lnBias: this.tensor( `${ prefix }.input_layernorm.bias` ),
			attnQKVWeight: packProjections( [ q, k, v ], hiddenSize ),
			attnQKVBias: packBiases( [
				this.bias( `${ prefix }.self_attn.q_proj.bias`, qSize ),
				this.bias( `${ prefix }.self_attn.k_proj.bias`, kvSize ),
				this.bias( `${ prefix }.self_attn.v_proj.bias`, kvSize )
			] ),
			attnProjWeight: this.linear( `${ prefix }.self_attn.dense.weight`, hiddenSize, qSize ),
			attnProjBias: this.bias( `${ prefix }.self_attn.dense.bias`, hiddenSize ),
			mlpFCWeight: this.linear( `${ prefix }.mlp.fc1.weight`, innerSize, hiddenSize ),
			mlpFCBias: this.bias( `${ prefix }.mlp.fc1.bias`, innerSize ),
			mlpProjWeight: this.linear( `${ prefix }.mlp.fc2.weight`, hiddenSize, innerSize ),
			mlpProjBias: this.bias( `${ prefix }.mlp.fc2.bias`, hiddenSize )
		};

	}

	embedding( tokenId, position, target = new Float32Array( this.hiddenSize ) ) {

		const tokenEmbedding = this.tensor( 'embed_tokens.weight' );
		const tokenOffset = tokenId * this.hiddenSize;
		target.set( tokenEmbedding.subarray( tokenOffset, tokenOffset + this.hiddenSize ) );

		return target;

	}

}

export { PhiWeights };

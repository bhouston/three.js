import { SafeTensorsLoader } from './SafeTensorsLoader.js';
import { GPT2Tokenizer } from './GPT2Tokenizer.js';
import { fetchJSON, prepareGeneration, tensorToFloat32, transpose2D } from './LLMTensors.js';

/**
 * Loads a tiny GPT-2 model directory exported in Hugging Face format.
 *
 * @three_import import { GPT2Weights } from 'three/addons/gpgpu/llm/GPT2Weights.js';
 */
class GPT2Weights {

	constructor( config, tensors, tokenizer ) {

		this.architecture = 'gpt2';
		this.config = config;
		this.tensors = tensors;
		this.tokenizer = tokenizer;
		this.tensorPrefix = tensors[ 'transformer.wte.weight' ] !== undefined ? 'transformer.' : '';
		this.hiddenSize = config.n_embd;
		this.innerSize = config.n_inner || config.n_embd * 4;
		this.layerCount = config.n_layer;
		this.headCount = config.n_head;
		this.kvHeadCount = config.n_head;
		this.headDim = config.n_embd / config.n_head;
		this.vocabSize = config.vocab_size;
		this.endOfTextTokenId = config.eos_token_id;
		this._float32 = new Map();

		this.logitWeight = transpose2D( this.tensor( 'wte.weight' ), this.vocabSize, this.hiddenSize );

	}

	contextLimit() {

		return this.config.n_positions || this.config.n_ctx || 1024;

	}

	/**
	 * Keep the prompt (right-truncated only if it exceeds the context window)
	 * and spend leftover slots on new tokens. Never shrink the prompt to make
	 * room for a requested generation length.
	 */
	prepareGeneration( prompt, maxTokens, maxNewTokens ) {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	static async fromURL( baseURL ) {

		const root = baseURL.endsWith( '/' ) ? baseURL : `${ baseURL }/`;
		const [ config, safeTensors, tokenizer ] = await Promise.all( [
			fetchJSON( `${ root }config.json`, 'GPT2Weights' ),
			new SafeTensorsLoader().load( `${ root }model.safetensors` ),
			GPT2Tokenizer.fromURLs( `${ root }vocab.json`, `${ root }merges.txt` )
		] );

		return new GPT2Weights( config, safeTensors.tensors, tokenizer );

	}

	tensor( name ) {

		const tensor = this.tensors[ `${ this.tensorPrefix }${ name }` ];

		if ( tensor === undefined ) {

			throw new Error( `GPT2Weights: Missing tensor "${ this.tensorPrefix }${ name }".` );

		}

		if ( this._float32.has( name ) ) return this._float32.get( name );

		const data = tensorToFloat32( tensor );
		this._float32.set( name, data );

		return data;

	}

	block( index ) {

		const prefix = `h.${ index }`;

		return {
			ln1Weight: this.tensor( `${ prefix }.ln_1.weight` ),
			ln1Bias: this.tensor( `${ prefix }.ln_1.bias` ),
			ln2Weight: this.tensor( `${ prefix }.ln_2.weight` ),
			ln2Bias: this.tensor( `${ prefix }.ln_2.bias` ),
			attnQKVWeight: this.tensor( `${ prefix }.attn.c_attn.weight` ),
			attnQKVBias: this.tensor( `${ prefix }.attn.c_attn.bias` ),
			attnProjWeight: this.tensor( `${ prefix }.attn.c_proj.weight` ),
			attnProjBias: this.tensor( `${ prefix }.attn.c_proj.bias` ),
			mlpFCWeight: this.tensor( `${ prefix }.mlp.c_fc.weight` ),
			mlpFCBias: this.tensor( `${ prefix }.mlp.c_fc.bias` ),
			mlpProjWeight: this.tensor( `${ prefix }.mlp.c_proj.weight` ),
			mlpProjBias: this.tensor( `${ prefix }.mlp.c_proj.bias` )
		};

	}

	embedding( tokenId, position, target = new Float32Array( this.hiddenSize ) ) {

		const tokenEmbedding = this.tensor( 'wte.weight' );
		const positionEmbedding = this.tensor( 'wpe.weight' );
		const tokenOffset = tokenId * this.hiddenSize;
		const positionOffset = position * this.hiddenSize;

		for ( let i = 0; i < this.hiddenSize; i ++ ) {

			target[ i ] = tokenEmbedding[ tokenOffset + i ] + positionEmbedding[ positionOffset + i ];

		}

		return target;

	}

}

export { GPT2Weights, transpose2D };

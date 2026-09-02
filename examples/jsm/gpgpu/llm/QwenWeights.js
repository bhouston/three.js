import { loadSafetensorsModel } from './SafeTensorsLoader.js';
import { GPT2Tokenizer, QWEN_TOKEN_PATTERN } from './GPT2Tokenizer.js';
import { fetchJSON, packProjections, prepareGeneration, tensorToFloat32, transpose2D, convertAllTensors, createProgress, unwrapTextConfig, detectLanguagePrefix } from './LLMTensors.js';

/**
 * Loads a Hugging Face Qwen3.5 text backbone (`qwen3_5` / `qwen3_5_text`).
 *
 * Hybrid decode: Gated DeltaNet linear-attention layers and gated full
 * attention with QK-norm and partial RoPE.
 *
 * @three_import import { QwenWeights } from 'three/addons/gpgpu/llm/QwenWeights.js';
 */
class QwenWeights {

	constructor( config, tensors, tokenizer, options = {} ) {

		this.architecture = 'qwen3_5';
		this.config = unwrapTextConfig( config );
		this.rawConfig = config;
		this.tensors = tensors;
		this.tokenizer = tokenizer;
		this.tensorPrefix = detectLanguagePrefix( tensors );
		this.hiddenSize = this.config.hidden_size;
		this.innerSize = this.config.intermediate_size;
		this.layerCount = this.config.num_hidden_layers;
		this.headCount = this.config.num_attention_heads;
		this.kvHeadCount = this.config.num_key_value_heads || this.config.num_attention_heads;
		this.headDim = this.config.head_dim || ( this.config.hidden_size / this.config.num_attention_heads );
		this.qSize = this.headCount * this.headDim;
		this.kvSize = this.kvHeadCount * this.headDim;
		this.vocabSize = this.config.vocab_size;
		this.layerTypes = this.config.layer_types || [];
		this.rmsNormEps = this.config.rms_norm_eps || 1e-6;
		this.offsetRMSNorm = true;
		this.mlpActivation = this.config.hidden_act || 'silu';
		this.linearKeyDim = this.config.linear_key_head_dim || 128;
		this.linearValueDim = this.config.linear_value_head_dim || 128;
		this.linearKeyHeads = this.config.linear_num_key_heads || 16;
		this.linearValueHeads = this.config.linear_num_value_heads || 16;
		this.linearConvKernel = this.config.linear_conv_kernel_dim || 4;
		this.ropeTheta = this.config.rope_parameters?.rope_theta || this.config.rope_theta || 10000000;
		this.partialRotaryFactor = this.config.rope_parameters?.partial_rotary_factor ?? 0.25;
		this.rotaryDim = Math.floor( this.headDim * this.partialRotaryFactor );
		this.attnScale = this.headDim ** - 0.5;
		this.endOfTextTokenId = Array.isArray( this.config.eos_token_id ) ? this.config.eos_token_id[ 0 ] : ( this.config.eos_token_id ?? tokenizer.endOfTextTokenId ?? 248044 );
		this._float32 = new Map();
		this.logitWeight = null;
		this._blocks = [];

		if ( options.deferUnpack !== true ) this.unpackSync();

	}

	lmHeadTensor() {

		const embedding = this.tensor( 'embed_tokens.weight' );
		return this.hasTensor( 'lm_head.weight' ) && this.config.tie_word_embeddings !== true
			? this.tensor( 'lm_head.weight' )
			: embedding;

	}

	unpackSync() {

		this.logitWeight = transpose2D( this.lmHeadTensor(), this.vocabSize, this.hiddenSize );

		for ( let i = 0; i < this.layerCount; i ++ ) this._blocks[ i ] = this.createBlock( i );

	}

	async unpack( onProgress ) {

		const report = createProgress( 'QwenWeights', onProgress );
		await report( `Transposing output projection (${ this.vocabSize } x ${ this.hiddenSize }); UI may pause...` );
		this.logitWeight = transpose2D( this.lmHeadTensor(), this.vocabSize, this.hiddenSize );

		for ( let i = 0; i < this.layerCount; i ++ ) {

			this._blocks[ i ] = this.createBlock( i );
			await report( `Unpacked layer ${ i + 1 } / ${ this.layerCount }` );

		}

	}

	contextLimit() {

		return Math.min( this.config.max_position_embeddings || 2048, 2048 );

	}

	prepareGeneration( prompt, maxTokens, maxNewTokens ) {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	static async fromURL( baseURL, options = {} ) {

		const root = baseURL.endsWith( '/' ) ? baseURL : `${ baseURL }/`;
		const report = createProgress( 'QwenWeights', options.onProgress );

		await report( `Loading config ${ root }config.json` );
		const config = await fetchJSON( `${ root }config.json`, 'QwenWeights' );
		const text = unwrapTextConfig( config );
		await report( `${ text.num_hidden_layers } layers, hidden ${ text.hidden_size }, vocab ${ text.vocab_size }` );
		await report( `Loading tokenizer ${ root }vocab.json` );
		const tokenizer = await GPT2Tokenizer.fromURLs( `${ root }vocab.json`, `${ root }merges.txt`, {
			tokenPattern: QWEN_TOKEN_PATTERN,
			endOfTextToken: '<|endoftext|>'
		} );
		const tensors = await loadSafetensorsModel( root, {
			onProgress: options.onProgress,
			label: 'QwenWeights',
			keepTensor: keepQwenTensor
		} );
		await convertAllTensors( tensors, options.onProgress, 'QwenWeights' );
		await report( 'Packing layers...' );
		const weights = new QwenWeights( config, tensors, tokenizer, { deferUnpack: true } );
		await weights.unpack( options.onProgress );
		await report( 'Weights ready' );
		return weights;

	}

	hasTensor( name ) {

		return this.tensors[ `${ this.tensorPrefix }${ name }` ] !== undefined || this.tensors[ name ] !== undefined;

	}

	tensor( name ) {

		if ( this._float32.has( name ) ) return this._float32.get( name );

		const tensor = this.tensors[ `${ this.tensorPrefix }${ name }` ] || this.tensors[ name ];

		if ( tensor === undefined ) {

			throw new Error( `QwenWeights: Missing tensor "${ this.tensorPrefix }${ name }".` );

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
		const layerType = this.layerTypes[ index ] || 'full_attention';
		const { hiddenSize, innerSize } = this;

		const block = {
			layerType,
			ln1Weight: this.tensor( `${ prefix }.input_layernorm.weight` ),
			ln2Weight: this.tensor( `${ prefix }.post_attention_layernorm.weight` ),
			mlpGateWeight: this.linearWeight( `${ prefix }.mlp.gate_proj.weight`, innerSize, hiddenSize ),
			mlpUpWeight: this.linearWeight( `${ prefix }.mlp.up_proj.weight`, innerSize, hiddenSize ),
			mlpDownWeight: this.linearWeight( `${ prefix }.mlp.down_proj.weight`, hiddenSize, innerSize )
		};

		if ( layerType === 'linear_attention' ) {

			const convDim = this.linearKeyHeads * this.linearKeyDim * 2 + this.linearValueHeads * this.linearValueDim;
			block.delta = {
				qkvWeight: this.linearWeight( `${ prefix }.linear_attn.in_proj_qkv.weight`, convDim, hiddenSize ),
				zWeight: this.linearWeight( `${ prefix }.linear_attn.in_proj_z.weight`, this.linearValueHeads * this.linearValueDim, hiddenSize ),
				bWeight: this.linearWeight( `${ prefix }.linear_attn.in_proj_b.weight`, this.linearValueHeads, hiddenSize ),
				aWeight: this.linearWeight( `${ prefix }.linear_attn.in_proj_a.weight`, this.linearValueHeads, hiddenSize ),
				outWeight: this.linearWeight( `${ prefix }.linear_attn.out_proj.weight`, hiddenSize, this.linearValueHeads * this.linearValueDim ),
				convWeight: this.tensor( `${ prefix }.linear_attn.conv1d.weight` ),
				aLog: this.tensor( `${ prefix }.linear_attn.A_log` ),
				dtBias: this.tensor( `${ prefix }.linear_attn.dt_bias` ),
				normWeight: this.tensor( `${ prefix }.linear_attn.norm.weight` )
			};

		} else {

			const qGate = this.linearWeight( `${ prefix }.self_attn.q_proj.weight`, this.qSize * 2, hiddenSize );
			const k = this.linearWeight( `${ prefix }.self_attn.k_proj.weight`, this.kvSize, hiddenSize );
			const v = this.linearWeight( `${ prefix }.self_attn.v_proj.weight`, this.kvSize, hiddenSize );
			block.qGateWeight = qGate;
			block.kWeight = k;
			block.vWeight = v;
			block.attnQKVWeight = packProjections( [ k, v ], hiddenSize );
			block.attnProjWeight = this.linearWeight( `${ prefix }.self_attn.o_proj.weight`, hiddenSize, this.qSize );
			block.qNormWeight = this.tensor( `${ prefix }.self_attn.q_norm.weight` );
			block.kNormWeight = this.tensor( `${ prefix }.self_attn.k_norm.weight` );

		}

		return block;

	}

	embedding( tokenId, position, target = new Float32Array( this.hiddenSize ) ) {

		const tokenEmbedding = this.tensor( 'embed_tokens.weight' );
		const tokenOffset = tokenId * this.hiddenSize;
		target.set( tokenEmbedding.subarray( tokenOffset, tokenOffset + this.hiddenSize ) );
		return target;

	}

}

function keepQwenTensor( name ) {

	return name.includes( 'language_model' ) && name.includes( 'visual' ) === false && name.startsWith( 'mtp.' ) === false;

}

export { QwenWeights };

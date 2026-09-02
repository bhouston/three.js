import { loadSafetensorsModel } from './SafeTensorsLoader.js';
import { UnigramTokenizer } from './UnigramTokenizer.js';
import { fetchJSON, packProjections, prepareGeneration, tensorToFloat32, transpose2D, convertAllTensors, createProgress, unwrapTextConfig, detectLanguagePrefix, bfloat16ToFloat32, float16ToFloat32 } from './LLMTensors.js';

/**
 * Loads a Hugging Face Gemma 4 text backbone (`gemma4` / `gemma4_text`).
 *
 * Adds per-layer embeddings, shared KV caches, double-wide MLPs on shared
 * layers, proportional RoPE on global attention, and logit softcapping.
 *
 * @three_import import { Gemma4Weights } from 'three/addons/gpgpu/llm/Gemma4Weights.js';
 */
class Gemma4Weights {

	constructor( config, tensors, tokenizer, options = {} ) {

		this.architecture = 'gemma4';
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
		this.localHeadDim = this.config.head_dim || 256;
		this.globalHeadDim = this.config.global_head_dim || 512;
		this.vocabSize = this.config.vocab_size;
		this.layerTypes = this.config.layer_types || [];
		this.slidingWindow = this.config.sliding_window || 512;
		this.rmsNormEps = this.config.rms_norm_eps || 1e-6;
		this.offsetRMSNorm = false;
		this.mlpActivation = this.config.hidden_activation || 'gelu_pytorch_tanh';
		this.embedScale = Math.sqrt( this.hiddenSize );
		this.attnScale = 1;
		this.finalLogitSoftcapping = this.config.final_logit_softcapping ?? 30;
		this.pleDim = this.config.hidden_size_per_layer_input || 0;
		this.useDoubleWideMLP = this.config.use_double_wide_mlp === true;
		this.numKvSharedLayers = this.config.num_kv_shared_layers || 0;
		this.firstKvSharedLayer = this.layerCount - this.numKvSharedLayers;
		this.sharedSource = lastNonSharedOfType( this.layerTypes, this.firstKvSharedLayer );
		this.ropeByType = this.config.rope_parameters || {
			sliding_attention: { rope_theta: 10000, rope_type: 'default' },
			full_attention: { rope_theta: 1000000, rope_type: 'proportional', partial_rotary_factor: 0.25 }
		};
		this.endOfTextTokenId = this.config.eos_token_id ?? tokenizer.endOfTextTokenId ?? 1;
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
		if ( this.pleDim > 0 ) this.unpackPLE();
		for ( let i = 0; i < this.layerCount; i ++ ) this._blocks[ i ] = this.createBlock( i );

	}

	async unpack( onProgress ) {

		const report = createProgress( 'Gemma4Weights', onProgress );
		await report( `Transposing output projection (${ this.vocabSize } x ${ this.hiddenSize }); UI may pause...` );
		this.logitWeight = transpose2D( this.lmHeadTensor(), this.vocabSize, this.hiddenSize );
		if ( this.pleDim > 0 ) {

			await report( 'Unpacking per-layer embeddings' );
			this.unpackPLE();

		}

		for ( let i = 0; i < this.layerCount; i ++ ) {

			this._blocks[ i ] = this.createBlock( i );
			await report( `Unpacked layer ${ i + 1 } / ${ this.layerCount }` );

		}

	}

	unpackPLE() {

		this.perLayerProjectionWeight = this.linearWeight(
			'per_layer_model_projection.weight',
			this.layerCount * this.pleDim,
			this.hiddenSize
		);
		this.perLayerProjectionNorm = this.tensor( 'per_layer_projection_norm.weight' );
		this.perLayerInputScale = 2 ** - 0.5;
		this.perLayerModelProjectionScale = this.hiddenSize ** - 0.5;
		this.pleEmbedScale = Math.sqrt( this.pleDim );

	}

	contextLimit() {

		return Math.min( this.config.max_position_embeddings || 512, 512 );

	}

	prepareGeneration( prompt, maxTokens, maxNewTokens ) {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	static async fromURL( baseURL, options = {} ) {

		const root = baseURL.endsWith( '/' ) ? baseURL : `${ baseURL }/`;
		const report = createProgress( 'Gemma4Weights', options.onProgress );

		await report( `Loading config ${ root }config.json` );
		const config = await fetchJSON( `${ root }config.json`, 'Gemma4Weights' );
		const text = unwrapTextConfig( config );
		await report( `${ text.num_hidden_layers } layers, hidden ${ text.hidden_size }, vocab ${ text.vocab_size }` );
		await report( `Loading tokenizer ${ root }tokenizer.json (this file is large)` );
		const tokenizer = await UnigramTokenizer.fromURL( root, options );
		const tensors = await loadSafetensorsModel( root, {
			onProgress: options.onProgress,
			label: 'Gemma4Weights',
			keepTensor: keepGemma4Tensor
		} );
		await convertAllTensors( tensors, options.onProgress, 'Gemma4Weights', skipGemma4Convert );
		const weights = new Gemma4Weights( config, tensors, tokenizer, { deferUnpack: true } );

		if ( text.bos_token_id !== undefined ) tokenizer.bosTokenId = text.bos_token_id;
		if ( text.eos_token_id !== undefined ) {

			tokenizer.eosTokenId = text.eos_token_id;
			tokenizer.endOfTextTokenId = text.eos_token_id;

		}

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

			throw new Error( `Gemma4Weights: Missing tensor "${ this.tensorPrefix }${ name }".` );

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

	layerHeadDim( layerType ) {

		return layerType === 'full_attention' ? this.globalHeadDim : this.localHeadDim;

	}

	createBlock( index ) {

		const prefix = `layers.${ index }`;
		const layerType = this.layerTypes[ index ] || 'sliding_attention';
		const headDim = this.layerHeadDim( layerType );
		const qSize = this.headCount * headDim;
		const kvSize = this.kvHeadCount * headDim;
		const isKVShared = index >= this.firstKvSharedLayer && this.firstKvSharedLayer >= 0;
		const innerSize = this.innerSize * ( this.useDoubleWideMLP && isKVShared ? 2 : 1 );
		const rope = this.ropeByType[ layerType ] || this.ropeByType.sliding_attention;
		const rotaryDim = headDim;
		const pairCount = rope.rope_type === 'proportional'
			? Math.floor( ( rope.partial_rotary_factor || 0.25 ) * headDim / 2 )
			: rotaryDim / 2;
		const q = this.linearWeight( `${ prefix }.self_attn.q_proj.weight`, qSize, this.hiddenSize );

		const block = {
			layerType,
			headDim,
			qSize,
			kvSize,
			innerSize,
			isKVShared,
			sharedSource: isKVShared ? this.sharedSource[ layerType ] : - 1,
			ropeTheta: rope.rope_theta || 10000,
			rotaryDim,
			ropeFreqDim: headDim,
			ropePairCount: pairCount,
			slidingWindow: layerType === 'sliding_attention' ? this.slidingWindow : 0,
			ln1Weight: this.tensor( `${ prefix }.input_layernorm.weight` ),
			postAttnNormWeight: this.tensor( `${ prefix }.post_attention_layernorm.weight` ),
			preMlpNormWeight: this.tensor( `${ prefix }.pre_feedforward_layernorm.weight` ),
			postMlpNormWeight: this.tensor( `${ prefix }.post_feedforward_layernorm.weight` ),
			qNormWeight: this.tensor( `${ prefix }.self_attn.q_norm.weight` ),
			attnProjWeight: this.linearWeight( `${ prefix }.self_attn.o_proj.weight`, this.hiddenSize, qSize ),
			mlpGateWeight: this.linearWeight( `${ prefix }.mlp.gate_proj.weight`, innerSize, this.hiddenSize ),
			mlpUpWeight: this.linearWeight( `${ prefix }.mlp.up_proj.weight`, innerSize, this.hiddenSize ),
			mlpDownWeight: this.linearWeight( `${ prefix }.mlp.down_proj.weight`, this.hiddenSize, innerSize ),
			layerScalar: this.hasTensor( `${ prefix }.layer_scalar` ) ? this.tensor( `${ prefix }.layer_scalar` )[ 0 ] : 1
		};

		if ( isKVShared === false ) {

			const k = this.linearWeight( `${ prefix }.self_attn.k_proj.weight`, kvSize, this.hiddenSize );
			const v = this.linearWeight( `${ prefix }.self_attn.v_proj.weight`, kvSize, this.hiddenSize );
			block.kNormWeight = this.tensor( `${ prefix }.self_attn.k_norm.weight` );
			block.attnQKVWeight = packProjections( [ q, k, v ], this.hiddenSize );

		} else {

			block.qWeight = q;

		}

		if ( this.pleDim > 0 ) {

			block.pleGateWeight = this.linearWeight( `${ prefix }.per_layer_input_gate.weight`, this.pleDim, this.hiddenSize );
			block.pleProjWeight = this.linearWeight( `${ prefix }.per_layer_projection.weight`, this.hiddenSize, this.pleDim );
			block.pleNormWeight = this.tensor( `${ prefix }.post_per_layer_input_norm.weight` );

		}

		return block;

	}

	embedding( tokenId, position, target = new Float32Array( this.hiddenSize ) ) {

		const tokenEmbedding = this.tensor( 'embed_tokens.weight' );
		const tokenOffset = tokenId * this.hiddenSize;

		for ( let i = 0; i < this.hiddenSize; i ++ ) {

			target[ i ] = tokenEmbedding[ tokenOffset + i ] * this.embedScale;

		}

		return target;

	}

	perLayerTokenEmbedding( tokenId ) {

		const width = this.layerCount * this.pleDim;
		const target = new Float32Array( width );
		const tensor = this.tensors[ `${ this.tensorPrefix }embed_tokens_per_layer.weight` ] || this.tensors[ 'embed_tokens_per_layer.weight' ];
		const offset = tokenId * width;
		const scale = this.pleEmbedScale;

		if ( tensor.dtype === 'F32' ) {

			for ( let i = 0; i < width; i ++ ) target[ i ] = tensor.data[ offset + i ] * scale;

		} else {

			const convert = tensor.dtype === 'BF16' ? bfloat16ToFloat32 : float16ToFloat32;

			for ( let i = 0; i < width; i ++ ) target[ i ] = convert( tensor.data[ offset + i ] ) * scale;

		}

		return target;

	}

}

function lastNonSharedOfType( layerTypes, firstShared ) {

	const sources = {};

	for ( let i = 0; i < firstShared; i ++ ) sources[ layerTypes[ i ] ] = i;

	return sources;

}

function keepGemma4Tensor( name ) {

	if ( name.includes( 'vision' ) || name.includes( 'audio' ) ) return false;
	return name.includes( 'language_model' ) || name.includes( 'embed_tokens' );

}

function skipGemma4Convert( name ) {

	return name.includes( 'embed_tokens_per_layer' );

}

export { Gemma4Weights };

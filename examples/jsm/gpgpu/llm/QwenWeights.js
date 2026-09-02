import { loadHFModelBundle } from './HFModelBundle.js';
import { packProjections, prepareGeneration, tensorToFloat32, transpose2D, unwrapTextConfig, createProgress } from './LLMTensors.js';
import { resolveTensor } from './TensorNameMap.js';
import { recipeFor } from './DecoderRecipe.js';

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
		this.recipe = recipeFor( config );
		this.tensors = tensors;
		this.tokenizer = tokenizer;
		this.tensorPrefix = options.prefix !== undefined ? options.prefix : detectPrefix( tensors );
		this.hiddenSize = this.recipe.hiddenSize;
		this.innerSize = this.recipe.innerSize;
		this.layerCount = this.recipe.layerCount;
		this.headCount = this.recipe.headCount;
		this.kvHeadCount = this.recipe.kvHeadCount;
		this.headDim = this.recipe.headDim;
		this.qSize = this.headCount * this.headDim;
		this.kvSize = this.kvHeadCount * this.headDim;
		this.vocabSize = this.recipe.vocabSize;
		this.layerTypes = this.recipe.layerTypes;
		this.rmsNormEps = this.recipe.normEps;
		this.offsetRMSNorm = true;
		this.mlpActivation = this.recipe.mlpActivation;
		this.linearKeyDim = this.recipe.linearKeyDim;
		this.linearValueDim = this.recipe.linearValueDim;
		this.linearKeyHeads = this.recipe.linearKeyHeads;
		this.linearValueHeads = this.recipe.linearValueHeads;
		this.linearConvKernel = this.recipe.linearConvKernel;
		this.ropeTheta = this.recipe.ropeTheta;
		this.partialRotaryFactor = this.config.rope_parameters?.partial_rotary_factor ?? 0.25;
		this.rotaryDim = this.recipe.rotaryDim;
		this.attnScale = this.recipe.attnScale;
		this.endOfTextTokenId = this.recipe.endOfTextTokenId ?? tokenizer.endOfTextTokenId ?? 248044;
		this.stopTokenIds = [ this.endOfTextTokenId ];
		this._float32 = new Map();

		const imEndTokenId = tokenizer.encoder?.[ '<|im_end|>' ];

		if ( imEndTokenId !== undefined && this.stopTokenIds.includes( imEndTokenId ) === false ) {

			this.stopTokenIds.push( imEndTokenId );

		}

		this.logitWeight = null;
		this._blocks = [];

		if ( options.deferUnpack !== true ) this.unpackSync();

	}

	contextLimit() {

		return this.recipe.contextLimit;

	}

	prepareGeneration( prompt, maxTokens, maxNewTokens ) {

		return prepareGeneration( this.tokenizer, prompt, maxTokens, maxNewTokens, this.endOfTextTokenId );

	}

	formatChat( messages, options = {} ) {

		const enableThinking = options.enableThinking === true;
		const addGenerationPrompt = options.addGenerationPrompt !== false;
		let prompt = '';

		for ( const message of messages ) {

			const role = message.role;
			const text = message.text ?? message.content ?? '';

			if ( role !== 'system' && role !== 'user' && role !== 'assistant' ) continue;

			prompt += `<|im_start|>${ role }\n${ text }<|im_end|>\n`;

		}

		if ( addGenerationPrompt ) {

			prompt += '<|im_start|>assistant\n';
			prompt += enableThinking ? '<think>\n' : '<think>\n\n</think>\n\n';

		}

		return prompt;

	}

	static async fromURL( baseURL, options = {} ) {

		const bundle = await loadHFModelBundle( baseURL, { ...options, label: 'QwenWeights' } );
		const weights = new QwenWeights( bundle.rawConfig, bundle.tensors, bundle.tokenizer, {
			deferUnpack: true,
			prefix: bundle.prefix
		} );
		await weights.unpack( options.onProgress );
		return weights;

	}

	unpackSync() {

		this.logitWeight = this.loadOutputWeight();
		for ( let i = 0; i < this.layerCount; i ++ ) this._blocks[ i ] = this.createBlock( i );

	}

	async unpack( onProgress ) {

		const report = createProgress( 'QwenWeights', onProgress );
		await report( `Transposing output projection (${ this.vocabSize } x ${ this.hiddenSize }); UI may pause...` );
		this.logitWeight = this.loadOutputWeight();

		for ( let i = 0; i < this.layerCount; i ++ ) {

			this._blocks[ i ] = this.createBlock( i );
			await report( `Unpacked layer ${ i + 1 } / ${ this.layerCount }` );

		}

	}

	hasTensor( name ) {

		return this.tensors[ `${ this.tensorPrefix }${ name }` ] !== undefined || this.tensors[ name ] !== undefined;

	}

	mappedFloat( key, bid ) {

		const cacheKey = bid === undefined ? key : `${ key }.${ bid }`;
		if ( this._float32.has( cacheKey ) ) return this._float32.get( cacheKey );
		const data = tensorToFloat32( resolveTensor( this.tensors, this.tensorPrefix, 'qwen3_5', key, bid ) );
		this._float32.set( cacheKey, data );
		return data;

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

	linearMapped( key, bid, outFeatures, inFeatures ) {

		return transpose2D( this.mappedFloat( key, bid ), outFeatures, inFeatures );

	}

	loadOutputWeight() {

		const source = this.hasTensor( 'lm_head.weight' ) && this.config.tie_word_embeddings !== true
			? this.mappedFloat( 'output' )
			: this.mappedFloat( 'token_embd' );
		return transpose2D( source, this.vocabSize, this.hiddenSize );

	}

	block( index ) {

		return this._blocks[ index ];

	}

	createBlock( index ) {

		const layerType = this.layerTypes[ index ] || 'full_attention';
		const { hiddenSize, innerSize } = this;
		const block = {
			layerType,
			ln1Weight: this.mappedFloat( 'attn_norm', index ),
			ln2Weight: this.mappedFloat( 'ffn_norm', index ),
			mlpGateWeight: this.linearMapped( 'ffn_gate', index, innerSize, hiddenSize ),
			mlpUpWeight: this.linearMapped( 'ffn_up', index, innerSize, hiddenSize ),
			mlpDownWeight: this.linearMapped( 'ffn_down', index, hiddenSize, innerSize )
		};

		if ( layerType === 'linear_attention' ) {

			const convDim = this.linearKeyHeads * this.linearKeyDim * 2 + this.linearValueHeads * this.linearValueDim;
			block.delta = {
				qkvWeight: this.linearMapped( 'delta_qkv', index, convDim, hiddenSize ),
				zWeight: this.linearMapped( 'delta_z', index, this.linearValueHeads * this.linearValueDim, hiddenSize ),
				bWeight: this.linearMapped( 'delta_b', index, this.linearValueHeads, hiddenSize ),
				aWeight: this.linearMapped( 'delta_a', index, this.linearValueHeads, hiddenSize ),
				outWeight: this.linearMapped( 'delta_out', index, hiddenSize, this.linearValueHeads * this.linearValueDim ),
				convWeight: this.mappedFloat( 'delta_conv', index ),
				aLog: this.mappedFloat( 'delta_a_log', index ),
				dtBias: this.mappedFloat( 'delta_dt_bias', index ),
				normWeight: this.mappedFloat( 'delta_norm', index )
			};

		} else {

			const qGate = this.linearMapped( 'attn_q', index, this.qSize * 2, hiddenSize );
			const k = this.linearMapped( 'attn_k', index, this.kvSize, hiddenSize );
			const v = this.linearMapped( 'attn_v', index, this.kvSize, hiddenSize );
			block.qGateWeight = qGate;
			block.kWeight = k;
			block.vWeight = v;
			block.attnQKVWeight = packProjections( [ k, v ], hiddenSize );
			block.attnProjWeight = this.linearMapped( 'attn_out', index, hiddenSize, this.qSize );
			block.qNormWeight = this.mappedFloat( 'attn_q_norm', index );
			block.kNormWeight = this.mappedFloat( 'attn_k_norm', index );

		}

		return block;

	}

	embedding( tokenId, position, target = new Float32Array( this.hiddenSize ) ) {

		const tokenEmbedding = this.mappedFloat( 'token_embd' );
		const tokenOffset = tokenId * this.hiddenSize;
		target.set( tokenEmbedding.subarray( tokenOffset, tokenOffset + this.hiddenSize ) );
		return target;

	}

}

function detectPrefix( tensors ) {

	if ( tensors[ 'model.language_model.embed_tokens.weight' ] !== undefined ) return 'model.language_model.';
	if ( tensors[ 'language_model.embed_tokens.weight' ] !== undefined ) return 'language_model.';
	if ( tensors[ 'model.embed_tokens.weight' ] !== undefined ) return 'model.';
	if ( tensors[ 'embed_tokens.weight' ] !== undefined ) return '';

	return 'model.';

}

export { QwenWeights };

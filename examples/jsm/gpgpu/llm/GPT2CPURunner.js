import { geluNew, layerNorm, linear, sampleTopK, softmax } from './LLMMath.js';

/**
 * CPU reference implementation of the toy GPT-2 forward pass.
 * Used to validate TSL kernels and lock greedy continuations.
 *
 * @three_import import { GPT2CPURunner } from 'three/addons/gpgpu/llm/GPT2CPURunner.js';
 */
class GPT2CPURunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.hiddenSize = weights.hiddenSize;
		this.caches = [];

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			this.caches.push( {
				key: new Float32Array( this.hiddenSize * this.maxTokens ),
				value: new Float32Array( this.hiddenSize * this.maxTokens )
			} );

		}

	}

	forwardToken( tokenId, position ) {

		const { weights, hiddenSize } = this;
		const x = weights.embedding( tokenId, position );

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const cache = this.caches[ i ];
			const norm1 = layerNorm( x, block.ln1Weight, block.ln1Bias, weights.config.layer_norm_epsilon );
			const qkv = linear( norm1, block.attnQKVWeight, block.attnQKVBias, hiddenSize, hiddenSize * 3 );
			const attn = attention( qkv, hiddenSize, weights.headCount, cache.key, cache.value, position );
			const attnOut = linear( attn, block.attnProjWeight, block.attnProjBias, hiddenSize, hiddenSize );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += attnOut[ dim ];

			const norm2 = layerNorm( x, block.ln2Weight, block.ln2Bias, weights.config.layer_norm_epsilon );
			const inner = linear( norm2, block.mlpFCWeight, block.mlpFCBias, hiddenSize, weights.innerSize );

			for ( let dim = 0; dim < inner.length; dim ++ ) inner[ dim ] = geluNew( inner[ dim ] );

			const mlpOut = linear( inner, block.mlpProjWeight, block.mlpProjBias, weights.innerSize, hiddenSize );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += mlpOut[ dim ];

		}

		const normed = layerNorm( x, weights.tensor( 'ln_f.weight' ), weights.tensor( 'ln_f.bias' ), weights.config.layer_norm_epsilon );

		return linear( normed, weights.logitWeight, null, hiddenSize, weights.vocabSize );

	}

	generate( prompt, options = {} ) {

		for ( const cache of this.caches ) {

			cache.key.fill( 0 );
			cache.value.fill( 0 );

		}

		const { inputTokens, newTokenBudget } = this.weights.prepareGeneration(
			prompt,
			this.maxTokens,
			options.maxNewTokens || 32
		);
		const allTokens = inputTokens.slice();
		const generatedTokens = [];
		let logits = null;

		for ( let i = 0; i < inputTokens.length; i ++ ) {

			logits = this.forwardToken( inputTokens[ i ], i );

		}

		for ( let i = 0; i < newTokenBudget; i ++ ) {

			const nextToken = sampleTopK( logits, { ...options, tokens: allTokens } );

			if ( nextToken === this.weights.endOfTextTokenId ) break;

			allTokens.push( nextToken );
			generatedTokens.push( nextToken );
			logits = this.forwardToken( nextToken, allTokens.length - 1 );

		}

		return {
			tokens: allTokens,
			generatedTokens,
			text: this.weights.tokenizer.decode( allTokens ),
			generatedText: this.weights.tokenizer.decode( generatedTokens )
		};

	}

}

function attention( qkv, hiddenSize, headCount, keyCache, valueCache, position ) {

	const headSize = hiddenSize / headCount;
	const scale = 1 / Math.sqrt( headSize );
	const output = new Float32Array( hiddenSize );

	for ( let dim = 0; dim < hiddenSize; dim ++ ) {

		keyCache[ position * hiddenSize + dim ] = qkv[ hiddenSize + dim ];
		valueCache[ position * hiddenSize + dim ] = qkv[ hiddenSize * 2 + dim ];

	}

	for ( let head = 0; head < headCount; head ++ ) {

		const headOffset = head * headSize;
		const scores = new Float32Array( position + 1 );

		for ( let token = 0; token <= position; token ++ ) {

			let dot = 0;

			for ( let i = 0; i < headSize; i ++ ) {

				dot += qkv[ headOffset + i ] * keyCache[ token * hiddenSize + headOffset + i ];

			}

			scores[ token ] = dot * scale;

		}

		const weights = softmax( scores );

		for ( let i = 0; i < headSize; i ++ ) {

			let sum = 0;

			for ( let token = 0; token <= position; token ++ ) {

				sum += weights[ token ] * valueCache[ token * hiddenSize + headOffset + i ];

			}

			output[ headOffset + i ] = sum;

		}

	}

	return output;

}

export { GPT2CPURunner };

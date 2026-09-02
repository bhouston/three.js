import { causalAttention, geluNew, layerNorm, linear, sampleTopK } from './LLMMath.js';

/**
 * CPU reference for Phi-1/Phi-2 decode (LayerNorm, partial RoPE, parallel MLP).
 *
 * @three_import import { PhiCPURunner } from 'three/addons/gpgpu/llm/PhiCPURunner.js';
 */
class PhiCPURunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.hiddenSize = weights.hiddenSize;
		this.caches = [];

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			this.caches.push( {
				key: new Float32Array( weights.kvSize * this.maxTokens ),
				value: new Float32Array( weights.kvSize * this.maxTokens )
			} );

		}

	}

	forwardToken( tokenId, position ) {

		const { weights, hiddenSize } = this;
		const x = weights.embedding( tokenId, position );

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const cache = this.caches[ i ];
			const normed = layerNorm( x, block.lnWeight, block.lnBias, weights.layerNormEps );
			const qkv = linear( normed, block.attnQKVWeight, block.attnQKVBias, hiddenSize, weights.qSize + 2 * weights.kvSize );
			const attn = causalAttention( qkv, {
				headCount: weights.headCount,
				kvHeadCount: weights.kvHeadCount,
				headDim: weights.headDim,
				position,
				keyCache: cache.key,
				valueCache: cache.value,
				ropeTheta: weights.ropeTheta,
				rotaryDim: weights.rotaryDim
			} );
			const attnOut = linear( attn, block.attnProjWeight, block.attnProjBias, weights.qSize, hiddenSize );
			const inner = linear( normed, block.mlpFCWeight, block.mlpFCBias, hiddenSize, weights.innerSize );

			for ( let dim = 0; dim < inner.length; dim ++ ) inner[ dim ] = geluNew( inner[ dim ] );

			const mlpOut = linear( inner, block.mlpProjWeight, block.mlpProjBias, weights.innerSize, hiddenSize );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += attnOut[ dim ] + mlpOut[ dim ];

		}

		const normed = layerNorm( x, weights.tensor( 'final_layernorm.weight' ), weights.tensor( 'final_layernorm.bias' ), weights.layerNormEps );

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

export { PhiCPURunner };

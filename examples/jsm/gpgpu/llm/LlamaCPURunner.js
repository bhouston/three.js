import { causalAttention, geluNew, linear, rmsNorm, sampleTopK, silu } from './LLMMath.js';

/**
 * CPU reference for Llama-style decode (RMSNorm, RoPE, GQA, SwiGLU).
 *
 * @three_import import { LlamaCPURunner } from 'three/addons/gpgpu/llm/LlamaCPURunner.js';
 */
class LlamaCPURunner {

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
			const norm1 = rmsNorm( x, block.ln1Weight, weights.rmsNormEps, weights.offsetRMSNorm );
			const qkv = linear( norm1, block.attnQKVWeight, block.attnQKVBias, hiddenSize, weights.qSize + 2 * weights.kvSize );
			const attn = causalAttention( qkv, {
				headCount: weights.headCount,
				kvHeadCount: weights.kvHeadCount,
				headDim: weights.headDim,
				position,
				keyCache: cache.key,
				valueCache: cache.value,
				ropeTheta: weights.ropeTheta,
				rotaryDim: weights.headDim
			} );
			const attnOut = linear( attn, block.attnProjWeight, block.attnProjBias, weights.qSize, hiddenSize );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += attnOut[ dim ];

			const norm2 = rmsNorm( x, block.ln2Weight, weights.rmsNormEps, weights.offsetRMSNorm );
			const gate = linear( norm2, block.mlpGateWeight, null, hiddenSize, weights.innerSize );
			const up = linear( norm2, block.mlpUpWeight, null, hiddenSize, weights.innerSize );
			const hidden = new Float32Array( weights.innerSize );
			const activate = weights.mlpActivation === 'silu' ? silu : geluNew;

			for ( let dim = 0; dim < hidden.length; dim ++ ) {

				hidden[ dim ] = activate( gate[ dim ] ) * up[ dim ];

			}

			const mlpOut = linear( hidden, block.mlpDownWeight, null, weights.innerSize, hiddenSize );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] += mlpOut[ dim ];

		}

		const normed = rmsNorm( x, weights.tensor( 'norm.weight' ), weights.rmsNormEps, weights.offsetRMSNorm );

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

			const nextToken = sampleTopK( logits, options );

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

export { LlamaCPURunner };

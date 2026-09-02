import { causalAttention, geluNew, linear, logitSoftcap, rmsNorm, sampleTopK } from './LLMMath.js';

/**
 * CPU reference for Gemma 4 decode.
 *
 * @three_import import { Gemma4CPURunner } from 'three/addons/gpgpu/llm/Gemma4CPURunner.js';
 */
class Gemma4CPURunner {

	constructor( weights, options = {} ) {

		this.weights = weights;
		this.maxTokens = Math.min( options.maxTokens || weights.contextLimit(), weights.contextLimit() );
		this.hiddenSize = weights.hiddenSize;
		this.caches = [];

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const headDim = weights.layerHeadDim( weights.layerTypes[ i ] || 'sliding_attention' );
			const kvSize = weights.kvHeadCount * headDim;
			this.caches.push( {
				key: new Float32Array( kvSize * this.maxTokens ),
				value: new Float32Array( kvSize * this.maxTokens )
			} );

		}

	}

	perLayerInputs( tokenId, embedding ) {

		const { weights } = this;
		if ( weights.pleDim <= 0 ) return null;

		const lookup = weights.perLayerTokenEmbedding( tokenId );
		const projected = linear( embedding, weights.perLayerProjectionWeight, null, weights.hiddenSize, weights.layerCount * weights.pleDim );

		for ( let i = 0; i < projected.length; i ++ ) projected[ i ] *= weights.perLayerModelProjectionScale;

		const combined = new Float32Array( projected.length );

		for ( let layer = 0; layer < weights.layerCount; layer ++ ) {

			const offset = layer * weights.pleDim;
			const slice = projected.subarray( offset, offset + weights.pleDim );
			const normed = rmsNorm( slice, weights.perLayerProjectionNorm, weights.rmsNormEps, false );

			for ( let i = 0; i < weights.pleDim; i ++ ) {

				combined[ offset + i ] = ( normed[ i ] + lookup[ offset + i ] ) * weights.perLayerInputScale;

			}

		}

		return combined;

	}

	forwardToken( tokenId, position ) {

		const { weights, hiddenSize } = this;
		const x = weights.embedding( tokenId, position );
		const ple = this.perLayerInputs( tokenId, x );

		for ( let i = 0; i < weights.layerCount; i ++ ) {

			const block = weights.block( i );
			const cache = block.isKVShared ? this.caches[ block.sharedSource ] : this.caches[ i ];
			const residualAttn = x.slice();
			const norm1 = rmsNorm( x, block.ln1Weight, weights.rmsNormEps, false );
			let qkv;

			if ( block.isKVShared ) {

				qkv = linear( norm1, block.qWeight, null, hiddenSize, block.qSize );

			} else {

				qkv = linear( norm1, block.attnQKVWeight, null, hiddenSize, block.qSize + 2 * block.kvSize );

			}

			const attn = causalAttention( qkv, {
				headCount: weights.headCount,
				kvHeadCount: weights.kvHeadCount,
				headDim: block.headDim,
				position,
				keyCache: cache.key,
				valueCache: cache.value,
				ropeTheta: block.ropeTheta,
				rotaryDim: block.rotaryDim,
				ropeFreqDim: block.ropeFreqDim,
				ropePairCount: block.ropePairCount,
				slidingWindow: block.slidingWindow,
				attnScale: weights.attnScale,
				qNormWeight: block.qNormWeight,
				kNormWeight: block.kNormWeight || null,
				rmsEpsilon: weights.rmsNormEps,
				offsetRMSNorm: false,
				queryOnly: block.isKVShared,
				writeCache: block.isKVShared === false,
				vNorm: block.isKVShared === false
			} );
			let attnOut = linear( attn, block.attnProjWeight, null, block.qSize, hiddenSize );
			attnOut = rmsNorm( attnOut, block.postAttnNormWeight, weights.rmsNormEps, false );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] = residualAttn[ dim ] + attnOut[ dim ];

			const residualMlp = x.slice();
			const preMlp = rmsNorm( x, block.preMlpNormWeight, weights.rmsNormEps, false );
			const gate = linear( preMlp, block.mlpGateWeight, null, hiddenSize, block.innerSize );
			const up = linear( preMlp, block.mlpUpWeight, null, hiddenSize, block.innerSize );
			const hidden = new Float32Array( block.innerSize );

			for ( let dim = 0; dim < hidden.length; dim ++ ) hidden[ dim ] = geluNew( gate[ dim ] ) * up[ dim ];

			let mlpOut = linear( hidden, block.mlpDownWeight, null, block.innerSize, hiddenSize );
			mlpOut = rmsNorm( mlpOut, block.postMlpNormWeight, weights.rmsNormEps, false );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] = residualMlp[ dim ] + mlpOut[ dim ];

			if ( ple !== null ) {

				const residualPle = x.slice();
				const pleGate = linear( x, block.pleGateWeight, null, hiddenSize, weights.pleDim );
				const pleHidden = new Float32Array( weights.pleDim );
				const pleOffset = i * weights.pleDim;

				for ( let dim = 0; dim < weights.pleDim; dim ++ ) pleHidden[ dim ] = geluNew( pleGate[ dim ] ) * ple[ pleOffset + dim ];

				let pleOut = linear( pleHidden, block.pleProjWeight, null, weights.pleDim, hiddenSize );
				pleOut = rmsNorm( pleOut, block.pleNormWeight, weights.rmsNormEps, false );

				for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] = residualPle[ dim ] + pleOut[ dim ];

			}

			if ( block.layerScalar !== 1 ) {

				for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] *= block.layerScalar;

			}

		}

		const normed = rmsNorm( x, weights.tensor( 'norm.weight' ), weights.rmsNormEps, false );
		const logits = linear( normed, weights.logitWeight, null, hiddenSize, weights.vocabSize );
		return logitSoftcap( logits, weights.finalLogitSoftcapping );

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

		for ( let i = 0; i < inputTokens.length; i ++ ) logits = this.forwardToken( inputTokens[ i ], i );

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

export { Gemma4CPURunner };

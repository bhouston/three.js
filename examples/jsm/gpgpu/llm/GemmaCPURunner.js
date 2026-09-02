import { causalAttention, geluNew, linear, rmsNorm } from './LLMMath.js';
import { generateSync } from './LLMGenerate.js';

/**
 * CPU reference for Gemma 3 decode.
 *
 * @three_import import { GemmaCPURunner } from 'three/addons/gpgpu/llm/GemmaCPURunner.js';
 */
class GemmaCPURunner {

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
			const residualAttn = x.slice();
			const norm1 = rmsNorm( x, block.ln1Weight, weights.rmsNormEps, true );
			const qkv = linear( norm1, block.attnQKVWeight, null, hiddenSize, weights.qSize + 2 * weights.kvSize );
			const attn = causalAttention( qkv, {
				headCount: weights.headCount,
				kvHeadCount: weights.kvHeadCount,
				headDim: weights.headDim,
				position,
				keyCache: cache.key,
				valueCache: cache.value,
				ropeTheta: block.ropeTheta,
				rotaryDim: weights.headDim,
				slidingWindow: block.slidingWindow,
				attnScale: weights.attnScale,
				qNormWeight: block.qNormWeight,
				kNormWeight: block.kNormWeight,
				rmsEpsilon: weights.rmsNormEps,
				offsetRMSNorm: true
			} );
			let attnOut = linear( attn, block.attnProjWeight, null, weights.qSize, hiddenSize );
			attnOut = rmsNorm( attnOut, block.postAttnNormWeight, weights.rmsNormEps, true );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] = residualAttn[ dim ] + attnOut[ dim ];

			const residualMlp = x.slice();
			const preMlp = rmsNorm( x, block.preMlpNormWeight, weights.rmsNormEps, true );
			const gate = linear( preMlp, block.mlpGateWeight, null, hiddenSize, weights.innerSize );
			const up = linear( preMlp, block.mlpUpWeight, null, hiddenSize, weights.innerSize );
			const hidden = new Float32Array( weights.innerSize );

			for ( let dim = 0; dim < hidden.length; dim ++ ) hidden[ dim ] = geluNew( gate[ dim ] ) * up[ dim ];

			let mlpOut = linear( hidden, block.mlpDownWeight, null, weights.innerSize, hiddenSize );
			mlpOut = rmsNorm( mlpOut, block.postMlpNormWeight, weights.rmsNormEps, true );

			for ( let dim = 0; dim < hiddenSize; dim ++ ) x[ dim ] = residualMlp[ dim ] + mlpOut[ dim ];

		}

		const normed = rmsNorm( x, weights.tensor( 'norm.weight' ), weights.rmsNormEps, true );

		return linear( normed, weights.logitWeight, null, hiddenSize, weights.vocabSize );

	}

	resetCache() {

		this._cacheTokens = [];
		this._cacheLogits = null;

		for ( const cache of this.caches ) {

			cache.key.fill( 0 );
			cache.value.fill( 0 );

		}

	}

	generate( prompt, options = {} ) {

		return generateSync( this, prompt, options, {
			rewindable: true,
			resetCache: () => this.resetCache(),
			forwardToken: ( tokenId, position ) => this.forwardToken( tokenId, position )
		} );

	}

}

export { GemmaCPURunner };

import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, Loop, cos, exp, float, instanceIndex, inversesqrt, pow, sin, storage, uint, uniform } from 'three/tsl';

/**
 * One-token causal self-attention for decode.
 *
 * Past keys/values stay in a cache. Each step writes the new K/V, then
 * materializes scaled Q·K scores once per (head, token), then applies the
 * same two-pass softmax-and-value mix as the CPU reference.
 *
 * GPT-2 uses packed QKV with one key/value head per query head. Llama-style
 * models can pass grouped-query (`kvHeadCount`) and rotary (`ropeTheta`)
 * options without changing the packed layout: `[Q, K, V]`.
 *
 * @three_import import { TSLAttention } from 'three/addons/gpgpu/llm/TSLAttention.js';
 */
class TSLAttention {

	constructor( qkvNode, hiddenSize, headCount, maxTokens, options = {} ) {

		this.qkvNode = qkvNode;
		this.hiddenSize = hiddenSize;
		this.headCount = headCount;
		this.headDim = options.headDim || hiddenSize / headCount;
		this.kvHeadCount = options.kvHeadCount || headCount;
		this.qSize = this.headCount * this.headDim;
		this.kvSize = this.kvHeadCount * this.headDim;
		this.maxTokens = maxTokens;
		this.workgroupSize = options.workgroupSize || 64;
		this.ropeTheta = options.ropeTheta || 0;
		this.rotaryDim = options.rotaryDim !== undefined ? options.rotaryDim : this.headDim;
		this.ropeFreqDim = options.ropeFreqDim || this.rotaryDim;
		this.ropePairCount = options.ropePairCount !== undefined ? options.ropePairCount : ( this.rotaryDim / 2 );
		this.slidingWindow = options.slidingWindow || 0;
		this.attnScale = options.attnScale !== undefined ? options.attnScale : ( 1 / Math.sqrt( this.headDim ) );
		this.rmsEpsilon = options.rmsEpsilon || 1e-6;
		this.offsetRMSNorm = options.offsetRMSNorm === true;
		this.vNorm = options.vNorm === true;
		this.sharedKV = options.sharedAttention !== undefined;
		this.gateNode = options.gateNode || null;
		this.qNormNode = null;
		this.kNormNode = null;
		this.position = uniform( 0, 'uint' );

		if ( options.qNormWeight ) {

			this.qNormAttribute = new StorageBufferAttribute( options.qNormWeight, 1 );
			this.qNormNode = storage( this.qNormAttribute, 'float', this.headDim ).toReadOnly().setName( options.name ? `${ options.name }QNorm` : 'LLMQNorm' );

		}

		if ( options.kNormWeight ) {

			this.kNormAttribute = new StorageBufferAttribute( options.kNormWeight, 1 );
			this.kNormNode = storage( this.kNormAttribute, 'float', this.headDim ).toReadOnly().setName( options.name ? `${ options.name }KNorm` : 'LLMKNorm' );

		}

		const cacheSize = this.kvSize * maxTokens;
		const scoreSize = headCount * maxTokens;
		const shared = options.sharedAttention;

		if ( shared ) {

			this.keyCacheAttribute = shared.keyCacheAttribute;
			this.valueCacheAttribute = shared.valueCacheAttribute;
			this.keyCacheNode = shared.keyCacheNode;
			this.valueCacheNode = shared.valueCacheNode;

		} else {

			this.keyCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
			this.valueCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
			this.keyCacheNode = storage( this.keyCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }KeyCache` : 'LLMKeyCache' );
			this.valueCacheNode = storage( this.valueCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }ValueCache` : 'LLMValueCache' );

		}

		this.scoreAttribute = new StorageBufferAttribute( new Float32Array( scoreSize ), 1 );
		this.queryAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );

		this.scoreNode = storage( this.scoreAttribute, 'float', scoreSize ).setName( options.name ? `${ options.name }Scores` : 'LLMAttentionScores' );
		this.queryNode = storage( this.queryAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Query` : 'LLMAttentionQuery' );
		this.outputNode = storage( this.outputAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Output` : 'LLMAttentionOutput' );

		this.copyComputeNode = this.sharedKV ? null : this.createCopyComputeNode( options.name ? `${ options.name }CopyKV` : 'LLMAttentionCopyKV' );
		this.queryComputeNode = this.createQueryComputeNode( options.name ? `${ options.name }Query` : 'LLMAttentionQuery' );
		this.scoreComputeNode = this.createScoreComputeNode( options.name ? `${ options.name }Scores` : 'LLMAttentionScores' );
		this.computeNode = this.createSoftmaxComputeNode( options.name || 'LLMAttention' );

	}

	headValue( vectorNode, headOffset, localDim, positionNode, normNode ) {

		const { headDim, rotaryDim, ropeTheta, rmsEpsilon, offsetRMSNorm, ropeFreqDim, ropePairCount } = this;
		const xRaw = vectorNode.element( headOffset.add( localDim ) );

		if ( ! normNode && ropeTheta <= 0 ) return xRaw;

		if ( ! normNode ) return this.ropeValue( vectorNode, headOffset, localDim, positionNode );

		const sumSquares = float( 0 ).toVar( 'sumSquares' );

		Loop( { start: uint( 0 ), end: uint( headDim ), type: 'uint', condition: '<' }, ( { i } ) => {

			const value = vectorNode.element( headOffset.add( i ) );
			sumSquares.addAssign( value.mul( value ) );

		} );

		const invRms = inversesqrt( sumSquares.div( float( headDim ) ).add( rmsEpsilon ) );
		const nScale = offsetRMSNorm ? normNode.element( localDim ).add( 1 ) : normNode.element( localDim );
		const x = xRaw.mul( invRms ).mul( nScale );

		if ( ropeTheta <= 0 || rotaryDim <= 0 ) return x;

		const half = uint( rotaryDim / 2 );
		const inRotary = localDim.lessThan( uint( rotaryDim ) );
		const freqIndex = localDim.mod( half );
		const partnerLocal = localDim.lessThan( half ).select( localDim.add( half ), localDim.sub( half ) );
		const partnerRaw = vectorNode.element( headOffset.add( partnerLocal ) );
		const partnerScale = offsetRMSNorm ? normNode.element( partnerLocal ).add( 1 ) : normNode.element( partnerLocal );
		const partner = localDim.lessThan( half ).select(
			partnerRaw.mul( invRms ).mul( partnerScale ).negate(),
			partnerRaw.mul( invRms ).mul( partnerScale )
		);
		const angle = freqIndex.lessThan( uint( ropePairCount ) ).select(
			float( positionNode ).mul( pow( float( ropeTheta ), float( freqIndex ).mul( float( - 2 / ropeFreqDim ) ) ) ),
			float( 0 )
		);
		const rotated = x.mul( cos( angle ) ).add( partner.mul( sin( angle ) ) );

		return inRotary.select( rotated, x );

	}

	ropeValue( vectorNode, headOffset, localDim, positionNode ) {

		const { rotaryDim, ropeTheta, qkvNode, ropeFreqDim, ropePairCount } = this;

		if ( ropeTheta <= 0 || rotaryDim <= 0 ) {

			return vectorNode.element( headOffset.add( localDim ) );

		}

		const half = uint( rotaryDim / 2 );
		const x = vectorNode.element( headOffset.add( localDim ) );
		const inRotary = localDim.lessThan( uint( rotaryDim ) );
		const freqIndex = localDim.mod( half );
		const partnerIndex = localDim.lessThan( half ).select(
			headOffset.add( localDim ).add( half ),
			headOffset.add( localDim ).sub( half )
		);
		const partner = localDim.lessThan( half ).select(
			qkvNode.element( partnerIndex ).negate(),
			qkvNode.element( partnerIndex )
		);
		const angle = freqIndex.lessThan( uint( ropePairCount ) ).select(
			float( positionNode ).mul( pow( float( ropeTheta ), float( freqIndex ).mul( float( - 2 / ropeFreqDim ) ) ) ),
			float( 0 )
		);
		const rotated = x.mul( cos( angle ) ).add( partner.mul( sin( angle ) ) );

		return inRotary.select( rotated, x );

	}

	createCopyComputeNode( name ) {

		const {
			qSize,
			kvSize,
			headDim,
			position,
			qkvNode,
			keyCacheNode,
			valueCacheNode,
			workgroupSize,
			ropeTheta
		} = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );

			If( dim.lessThan( uint( kvSize ) ), () => {

				const cacheOffset = position.mul( uint( kvSize ) ).add( dim );
				const headOffset = dim.div( uint( headDim ) ).mul( uint( headDim ) ).add( uint( qSize ) );
				const localDim = dim.mod( uint( headDim ) );
				const key = this.headValue( qkvNode, headOffset, localDim, position, this.kNormNode );
				let value = qkvNode.element( uint( qSize + kvSize ).add( dim ) );

				if ( this.vNorm ) {

					const valueHead = dim.div( uint( headDim ) ).mul( uint( headDim ) ).add( uint( qSize + kvSize ) );
					const sumSquares = float( 0 ).toVar( 'valueSumSquares' );

					Loop( { start: uint( 0 ), end: uint( headDim ), type: 'uint', condition: '<' }, ( { i } ) => {

						const sample = qkvNode.element( valueHead.add( i ) );
						sumSquares.addAssign( sample.mul( sample ) );

					} );

					value = value.mul( inversesqrt( sumSquares.div( float( headDim ) ).add( this.rmsEpsilon ) ) );

				}

				keyCacheNode.element( cacheOffset ).assign( key );
				valueCacheNode.element( cacheOffset ).assign( value );

			} );

		} )().compute( kvSize, [ workgroupSize ] ).setName( name );

	}

	createQueryComputeNode( name ) {

		const {
			qSize,
			headDim,
			position,
			qkvNode,
			queryNode,
			workgroupSize,
			ropeTheta
		} = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );

			If( dim.lessThan( uint( qSize ) ), () => {

				const headOffset = dim.div( uint( headDim ) ).mul( uint( headDim ) );
				const localDim = dim.mod( uint( headDim ) );
				const query = this.headValue( qkvNode, headOffset, localDim, position, this.qNormNode );

				queryNode.element( dim ).assign( query );

			} );

		} )().compute( qSize, [ workgroupSize ] ).setName( name );

	}

	createScoreComputeNode( name ) {

		const {
			headDim,
			headCount,
			kvHeadCount,
			kvSize,
			maxTokens,
			position,
			queryNode,
			keyCacheNode,
			scoreNode,
			workgroupSize,
			slidingWindow
		} = this;

		const scoreCount = headCount * maxTokens;
		const scale = this.attnScale;

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( scoreCount ) ), () => {

				const head = index.div( uint( maxTokens ) );
				const token = index.mod( uint( maxTokens ) );
				const windowStart = slidingWindow > 0
					? position.add( uint( 1 ) ).lessThan( uint( slidingWindow ) ).select( uint( 0 ), position.add( uint( 1 ) ).sub( uint( slidingWindow ) ) )
					: uint( 0 );

				If( token.greaterThanEqual( windowStart ).and( token.lessThan( position.add( uint( 1 ) ) ) ), () => {

					const qOffset = head.mul( uint( headDim ) );
					const kvHead = head.mul( uint( kvHeadCount ) ).div( uint( headCount ) );
					const kvOffset = kvHead.mul( uint( headDim ) );
					const score = float( 0 ).toVar( 'score' );

					Loop( { start: uint( 0 ), end: uint( headDim ), type: 'uint', condition: '<' }, ( { i } ) => {

						score.addAssign( queryNode.element( qOffset.add( i ) ).mul( keyCacheNode.element( token.mul( uint( kvSize ) ).add( kvOffset ).add( i ) ) ) );

					} );

					scoreNode.element( index ).assign( score.mul( scale ) );

				} );

			} );

		} )().compute( scoreCount, [ workgroupSize ] ).setName( name );

	}

	createSoftmaxComputeNode( name ) {

		const {
			qSize,
			headDim,
			headCount,
			kvHeadCount,
			kvSize,
			maxTokens,
			position,
			valueCacheNode,
			scoreNode,
			outputNode,
			workgroupSize,
			slidingWindow
		} = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );

			If( dim.lessThan( uint( qSize ) ), () => {

				const head = dim.div( uint( headDim ) );
				const localDim = dim.mod( uint( headDim ) );
				const kvHead = head.mul( uint( kvHeadCount ) ).div( uint( headCount ) );
				const kvOffset = kvHead.mul( uint( headDim ) );
				const scoreOffset = head.mul( uint( maxTokens ) );
				const windowStart = slidingWindow > 0
					? position.add( uint( 1 ) ).lessThan( uint( slidingWindow ) ).select( uint( 0 ), position.add( uint( 1 ) ).sub( uint( slidingWindow ) ) )
					: uint( 0 );
				const maxScore = float( - 3.4028234663852886e38 ).toVar( 'maxScore' );

				Loop( { start: windowStart, end: position.add( uint( 1 ) ), type: 'uint', condition: '<', name: 'token' }, ( { token } ) => {

					const score = scoreNode.element( scoreOffset.add( token ) );

					If( score.greaterThan( maxScore ), () => {

						maxScore.assign( score );

					} );

				} );

				const denominator = float( 0 ).toVar( 'denominator' );
				const value = float( 0 ).toVar( 'value' );

				Loop( { start: windowStart, end: position.add( uint( 1 ) ), type: 'uint', condition: '<', name: 'token' }, ( { token } ) => {

					const probability = exp( scoreNode.element( scoreOffset.add( token ) ).sub( maxScore ) );
					const v = valueCacheNode.element( token.mul( uint( kvSize ) ).add( kvOffset ).add( localDim ) );

					denominator.addAssign( probability );
					value.addAssign( probability.mul( v ) );

				} );

				outputNode.element( dim ).assign( this.gateNode
					? value.div( denominator ).mul( float( 1 ).div( float( 1 ).add( exp( this.gateNode.element( dim ).negate() ) ) ) )
					: value.div( denominator ) );

			} );

		} )().compute( qSize, [ workgroupSize ] ).setName( name );

	}

	setPosition( position ) {

		this.position.value = position;

	}

	reset() {

		if ( this.sharedKV ) return;

		this.keyCacheAttribute.array.fill( 0 );
		this.valueCacheAttribute.array.fill( 0 );
		this.keyCacheAttribute.needsUpdate = true;
		this.valueCacheAttribute.needsUpdate = true;

	}

	compute( renderer, position ) {

		this.setPosition( position );
		if ( this.copyComputeNode ) renderer.compute( this.copyComputeNode );
		renderer.compute( this.queryComputeNode );
		renderer.compute( this.scoreComputeNode );
		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLAttention };

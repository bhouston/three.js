import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, Loop, cos, exp, float, instanceIndex, pow, sin, storage, uint, uniform } from 'three/tsl';

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
		this.slidingWindow = options.slidingWindow || 0;
		this.position = uniform( 0, 'uint' );

		const cacheSize = this.kvSize * maxTokens;
		const scoreSize = headCount * maxTokens;

		this.keyCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
		this.valueCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
		this.scoreAttribute = new StorageBufferAttribute( new Float32Array( scoreSize ), 1 );
		this.queryAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( this.qSize ), 1 );

		this.keyCacheNode = storage( this.keyCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }KeyCache` : 'LLMKeyCache' );
		this.valueCacheNode = storage( this.valueCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }ValueCache` : 'LLMValueCache' );
		this.scoreNode = storage( this.scoreAttribute, 'float', scoreSize ).setName( options.name ? `${ options.name }Scores` : 'LLMAttentionScores' );
		this.queryNode = storage( this.queryAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Query` : 'LLMAttentionQuery' );
		this.outputNode = storage( this.outputAttribute, 'float', this.qSize ).setName( options.name ? `${ options.name }Output` : 'LLMAttentionOutput' );

		this.copyComputeNode = this.createCopyComputeNode( options.name ? `${ options.name }CopyKV` : 'LLMAttentionCopyKV' );
		this.queryComputeNode = this.createQueryComputeNode( options.name ? `${ options.name }Query` : 'LLMAttentionQuery' );
		this.scoreComputeNode = this.createScoreComputeNode( options.name ? `${ options.name }Scores` : 'LLMAttentionScores' );
		this.computeNode = this.createSoftmaxComputeNode( options.name || 'LLMAttention' );

	}

	ropeValue( vectorNode, headOffset, localDim, positionNode ) {

		const { rotaryDim, ropeTheta, qkvNode } = this;

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
		const angle = float( positionNode ).mul( pow( float( ropeTheta ), float( freqIndex ).mul( float( - 2 / rotaryDim ) ) ) );
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
				const key = ropeTheta > 0
					? this.ropeValue( qkvNode, headOffset, localDim, position )
					: qkvNode.element( uint( qSize ).add( dim ) );

				keyCacheNode.element( cacheOffset ).assign( key );
				valueCacheNode.element( cacheOffset ).assign( qkvNode.element( uint( qSize + kvSize ).add( dim ) ) );

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
				const query = ropeTheta > 0
					? this.ropeValue( qkvNode, headOffset, localDim, position )
					: qkvNode.element( dim );

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
		const scale = 1 / Math.sqrt( headDim );

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

				outputNode.element( dim ).assign( value.div( denominator ) );

			} );

		} )().compute( qSize, [ workgroupSize ] ).setName( name );

	}

	setPosition( position ) {

		this.position.value = position;

	}

	compute( renderer, position ) {

		this.setPosition( position );
		renderer.compute( this.copyComputeNode );
		renderer.compute( this.queryComputeNode );
		renderer.compute( this.scoreComputeNode );
		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLAttention };

import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, Loop, exp, float, instanceIndex, storage, uint, uniform } from 'three/tsl';

/**
 * One-token causal self-attention for GPT-2 decode.
 *
 * Past keys/values stay in a cache. Each step writes the new K/V, then
 * materializes scaled Q·K scores once per (head, token), then applies the
 * same two-pass softmax-and-value mix as the CPU reference.
 *
 * @three_import import { TSLAttention } from 'three/addons/gpgpu/llm/TSLAttention.js';
 */
class TSLAttention {

	constructor( qkvNode, hiddenSize, headCount, maxTokens, options = {} ) {

		this.hiddenSize = hiddenSize;
		this.headCount = headCount;
		this.headSize = hiddenSize / headCount;
		this.maxTokens = maxTokens;
		this.workgroupSize = options.workgroupSize || 64;
		this.position = uniform( 0, 'uint' );

		const cacheSize = hiddenSize * maxTokens;
		const scoreSize = headCount * maxTokens;

		this.keyCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
		this.valueCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
		this.scoreAttribute = new StorageBufferAttribute( new Float32Array( scoreSize ), 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( hiddenSize ), 1 );

		this.keyCacheNode = storage( this.keyCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }KeyCache` : 'LLMKeyCache' );
		this.valueCacheNode = storage( this.valueCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }ValueCache` : 'LLMValueCache' );
		this.scoreNode = storage( this.scoreAttribute, 'float', scoreSize ).setName( options.name ? `${ options.name }Scores` : 'LLMAttentionScores' );
		this.outputNode = storage( this.outputAttribute, 'float', hiddenSize ).setName( options.name ? `${ options.name }Output` : 'LLMAttentionOutput' );

		this.copyComputeNode = this.createCopyComputeNode( qkvNode, options.name ? `${ options.name }CopyKV` : 'LLMAttentionCopyKV' );
		this.scoreComputeNode = this.createScoreComputeNode( qkvNode, options.name ? `${ options.name }Scores` : 'LLMAttentionScores' );
		this.computeNode = this.createSoftmaxComputeNode( options.name || 'LLMAttention' );

	}

	createCopyComputeNode( qkvNode, name ) {

		const {
			hiddenSize,
			position,
			keyCacheNode,
			valueCacheNode,
			workgroupSize
		} = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );

			If( dim.lessThan( uint( hiddenSize ) ), () => {

				const cacheOffset = position.mul( uint( hiddenSize ) ).add( dim );

				keyCacheNode.element( cacheOffset ).assign( qkvNode.element( uint( hiddenSize ).add( dim ) ) );
				valueCacheNode.element( cacheOffset ).assign( qkvNode.element( uint( hiddenSize * 2 ).add( dim ) ) );

			} );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	createScoreComputeNode( qkvNode, name ) {

		const {
			hiddenSize,
			headSize,
			headCount,
			maxTokens,
			position,
			keyCacheNode,
			scoreNode,
			workgroupSize
		} = this;

		const scoreCount = headCount * maxTokens;
		const scale = 1 / Math.sqrt( headSize );

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( scoreCount ) ), () => {

				const head = index.div( uint( maxTokens ) );
				const token = index.mod( uint( maxTokens ) );

				If( token.lessThan( position.add( uint( 1 ) ) ), () => {

					const headOffset = head.mul( uint( headSize ) );
					const score = float( 0 ).toVar( 'score' );

					Loop( { start: uint( 0 ), end: uint( headSize ), type: 'uint', condition: '<' }, ( { i } ) => {

						const channel = headOffset.add( i );

						score.addAssign( qkvNode.element( channel ).mul( keyCacheNode.element( token.mul( uint( hiddenSize ) ).add( channel ) ) ) );

					} );

					scoreNode.element( index ).assign( score.mul( scale ) );

				} );

			} );

		} )().compute( scoreCount, [ workgroupSize ] ).setName( name );

	}

	createSoftmaxComputeNode( name ) {

		const {
			hiddenSize,
			headSize,
			maxTokens,
			position,
			valueCacheNode,
			scoreNode,
			outputNode,
			workgroupSize
		} = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );

			If( dim.lessThan( uint( hiddenSize ) ), () => {

				const head = dim.div( uint( headSize ) );
				const localDim = dim.mod( uint( headSize ) );
				const headOffset = head.mul( uint( headSize ) );
				const scoreOffset = head.mul( uint( maxTokens ) );
				const maxScore = float( - 3.4028234663852886e38 ).toVar( 'maxScore' );

				Loop( { start: uint( 0 ), end: position.add( uint( 1 ) ), type: 'uint', condition: '<', name: 'token' }, ( { token } ) => {

					const score = scoreNode.element( scoreOffset.add( token ) );

					If( score.greaterThan( maxScore ), () => {

						maxScore.assign( score );

					} );

				} );

				const denominator = float( 0 ).toVar( 'denominator' );
				const value = float( 0 ).toVar( 'value' );

				Loop( { start: uint( 0 ), end: position.add( uint( 1 ) ), type: 'uint', condition: '<', name: 'token' }, ( { token } ) => {

					const probability = exp( scoreNode.element( scoreOffset.add( token ) ).sub( maxScore ) );
					const v = valueCacheNode.element( token.mul( uint( hiddenSize ) ).add( headOffset ).add( localDim ) );

					denominator.addAssign( probability );
					value.addAssign( probability.mul( v ) );

				} );

				outputNode.element( dim ).assign( value.div( denominator ) );

			} );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	setPosition( position ) {

		this.position.value = position;

	}

	compute( renderer, position ) {

		this.setPosition( position );
		renderer.compute( this.copyComputeNode );
		renderer.compute( this.scoreComputeNode );
		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLAttention };

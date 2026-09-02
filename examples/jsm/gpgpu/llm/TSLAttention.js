import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, Loop, exp, float, instanceIndex, storage, uint, uniform } from 'three/tsl';

/**
 * Naive one-token causal self-attention for GPT-2.
 *
 * The pass appends the current token's K/V vectors to cache buffers and
 * computes the attention output for that token. It is intentionally simple:
 * one invocation per hidden dimension, with scalar loops over sequence length
 * and head dimension.
 *
 * @three_import import { TSLAttention } from 'three/addons/gpgpu/llm/TSLAttention.js';
 */
class TSLAttention {

	constructor( qkvNode, hiddenSize, headCount, maxTokens, options = {} ) {

		this.hiddenSize = hiddenSize;
		this.headCount = headCount;
		this.headSize = hiddenSize / headCount;
		this.maxTokens = maxTokens;
		this.position = uniform( 0, 'uint' );

		const cacheSize = hiddenSize * maxTokens;

		this.keyCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
		this.valueCacheAttribute = new StorageBufferAttribute( new Float32Array( cacheSize ), 1 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( hiddenSize ), 1 );

		this.keyCacheNode = storage( this.keyCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }KeyCache` : 'LLMKeyCache' );
		this.valueCacheNode = storage( this.valueCacheAttribute, 'float', cacheSize ).setName( options.name ? `${ options.name }ValueCache` : 'LLMValueCache' );
		this.outputNode = storage( this.outputAttribute, 'float', hiddenSize ).setName( options.name ? `${ options.name }Output` : 'LLMAttentionOutput' );

		this.copyComputeNode = this.createCopyComputeNode( qkvNode, options.name ? `${ options.name }CopyKV` : 'LLMAttentionCopyKV', options.workgroupSize || 64 );
		this.computeNode = this.createComputeNode( qkvNode, options.name || 'LLMAttention', options.workgroupSize || 64 );

	}

	createCopyComputeNode( qkvNode, name, workgroupSize ) {

		const {
			hiddenSize,
			position,
			keyCacheNode,
			valueCacheNode
		} = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );
			const cacheOffset = position.mul( uint( hiddenSize ) ).add( dim );

			keyCacheNode.element( cacheOffset ).assign( qkvNode.element( uint( hiddenSize ).add( dim ) ) );
			valueCacheNode.element( cacheOffset ).assign( qkvNode.element( uint( hiddenSize * 2 ).add( dim ) ) );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	createComputeNode( qkvNode, name, workgroupSize ) {

		const {
			hiddenSize,
			headSize,
			position,
			keyCacheNode,
			valueCacheNode,
			outputNode
		} = this;

		return Fn( () => {

			const dim = instanceIndex.toVar( 'dim' );
			const head = dim.div( uint( headSize ) );
			const localDim = dim.mod( uint( headSize ) );
			const headOffset = head.mul( uint( headSize ) );
			const maxScore = float( - 3.4028234663852886e38 ).toVar( 'maxScore' );
			const scale = float( 1 / Math.sqrt( headSize ) );

			Loop( { start: uint( 0 ), end: position.add( uint( 1 ) ), type: 'uint', condition: '<', name: 'token' }, ( { token } ) => {

				const score = float( 0 ).toVar( 'score' );

				Loop( { start: uint( 0 ), end: uint( headSize ), type: 'uint', condition: '<' }, ( { i } ) => {

					const channel = headOffset.add( i );
					const q = qkvNode.element( channel );
					const k = keyCacheNode.element( token.mul( uint( hiddenSize ) ).add( channel ) );

					score.addAssign( q.mul( k ) );

				} );

				score.mulAssign( scale );

				If( score.greaterThan( maxScore ), () => {

					maxScore.assign( score );

				} );

			} );

			const denominator = float( 0 ).toVar( 'denominator' );

			Loop( { start: uint( 0 ), end: position.add( uint( 1 ) ), type: 'uint', condition: '<', name: 'token' }, ( { token } ) => {

				const score = float( 0 ).toVar( 'score' );

				Loop( { start: uint( 0 ), end: uint( headSize ), type: 'uint', condition: '<' }, ( { i } ) => {

					const channel = headOffset.add( i );
					const q = qkvNode.element( channel );
					const k = keyCacheNode.element( token.mul( uint( hiddenSize ) ).add( channel ) );

					score.addAssign( q.mul( k ) );

				} );

				denominator.addAssign( exp( score.mul( scale ).sub( maxScore ) ) );

			} );

			const value = float( 0 ).toVar( 'value' );

			Loop( { start: uint( 0 ), end: position.add( uint( 1 ) ), type: 'uint', condition: '<', name: 'token' }, ( { token } ) => {

				const score = float( 0 ).toVar( 'score' );

				Loop( { start: uint( 0 ), end: uint( headSize ), type: 'uint', condition: '<' }, ( { i } ) => {

					const channel = headOffset.add( i );
					const q = qkvNode.element( channel );
					const k = keyCacheNode.element( token.mul( uint( hiddenSize ) ).add( channel ) );

					score.addAssign( q.mul( k ) );

				} );

				const probability = exp( score.mul( scale ).sub( maxScore ) ).div( denominator );
				const v = valueCacheNode.element( token.mul( uint( hiddenSize ) ).add( headOffset ).add( localDim ) );

				value.addAssign( probability.mul( v ) );

			} );

			outputNode.element( dim ).assign( value );

		} )().compute( hiddenSize, [ workgroupSize ] ).setName( name );

	}

	setPosition( position ) {

		this.position.value = position;

	}

	compute( renderer, position ) {

		this.setPosition( position );
		renderer.compute( this.copyComputeNode );
		renderer.compute( this.computeNode );

		return this.outputNode;

	}

}

export { TSLAttention };

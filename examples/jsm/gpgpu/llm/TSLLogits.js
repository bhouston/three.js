import { StorageBufferAttribute } from 'three/webgpu';
import {
	Fn, If, Loop, float, instanceIndex, invocationLocalIndex, storage, tanh, uint,
	workgroupArray, workgroupBarrier, workgroupId
} from 'three/tsl';

import { sampleTopKCandidates } from './LLMMath.js';
import { TSLLinear } from './TSLLinear.js';

const LOWEST_FLOAT = - 3.4028234663852886e38;

function createChunkedLogitLayers( inputNode, weights, chunkSize, name ) {

	const logits = [];
	const hiddenSize = weights.hiddenSize;

	for ( let offset = 0; offset < weights.vocabSize; offset += chunkSize ) {

		const size = Math.min( chunkSize, weights.vocabSize - offset );
		const chunkWeight = new Float32Array( hiddenSize * size );

		for ( let i = 0; i < hiddenSize; i ++ ) {

			const sourceOffset = i * weights.vocabSize + offset;
			chunkWeight.set( weights.logitWeight.subarray( sourceOffset, sourceOffset + size ), i * size );

		}

		logits.push( {
			offset,
			size,
			layer: new TSLLinear( inputNode, chunkWeight, null, hiddenSize, size, {
				name: `${ name }${ offset }`,
				workgroupSize: 256
			} )
		} );

	}

	return logits;

}

async function readChunkedLogits( renderer, chunks, vocabSize ) {

	const logits = new Float32Array( vocabSize );
	const values = await Promise.all( chunks.map( async ( chunk ) => (
		new Float32Array( await renderer.getArrayBufferAsync( chunk.layer.outputAttribute ) )
	) ) );

	for ( let i = 0; i < chunks.length; i ++ ) {

		const chunk = chunks[ i ];
		logits.set( values[ i ].subarray( 0, chunk.size ), chunk.offset );

	}

	return logits;

}

class TSLLogitSampler {

	constructor( chunks, options = {} ) {

		this.chunks = chunks;
		this.candidateCount = Math.max( 1, options.candidateCount || 8 );
		this.workgroupSize = options.workgroupSize || 256;
		this.logitSoftcap = options.logitSoftcap;
		this.chunkCandidateCount = chunks.length * this.candidateCount;
		this.partialOffsets = [];
		this.partialCount = 0;

		for ( const chunk of chunks ) {

			this.partialOffsets.push( this.partialCount );
			this.partialCount += Math.ceil( chunk.size / this.workgroupSize );

		}

		this.partialTokenAttribute = new StorageBufferAttribute( new Uint32Array( this.partialCount ), 1 );
		this.partialScoreAttribute = new StorageBufferAttribute( new Float32Array( this.partialCount ), 1 );
		this.chunkCandidateTokenAttribute = new StorageBufferAttribute( new Uint32Array( this.chunkCandidateCount ), 1 );
		this.chunkCandidateScoreAttribute = new StorageBufferAttribute( new Float32Array( this.chunkCandidateCount ), 1 );
		this.candidateTokenAttribute = new StorageBufferAttribute( new Uint32Array( this.candidateCount ), 1 );
		this.candidateScoreAttribute = new StorageBufferAttribute( new Float32Array( this.candidateCount ), 1 );
		this.partialTokenNode = storage( this.partialTokenAttribute, 'uint', this.partialCount ).setName( options.name ? `${ options.name }PartialTokens` : 'LLMPartialTokens' );
		this.partialScoreNode = storage( this.partialScoreAttribute, 'float', this.partialCount ).setName( options.name ? `${ options.name }PartialScores` : 'LLMPartialScores' );
		this.chunkCandidateTokenNode = storage( this.chunkCandidateTokenAttribute, 'uint', this.chunkCandidateCount ).setName( options.name ? `${ options.name }ChunkTokenCandidates` : 'LLMChunkTokenCandidates' );
		this.chunkCandidateScoreNode = storage( this.chunkCandidateScoreAttribute, 'float', this.chunkCandidateCount ).setName( options.name ? `${ options.name }ChunkScoreCandidates` : 'LLMChunkScoreCandidates' );
		this.candidateTokenNode = storage( this.candidateTokenAttribute, 'uint', this.candidateCount ).setName( options.name ? `${ options.name }TokenCandidates` : 'LLMTokenCandidates' );
		this.candidateScoreNode = storage( this.candidateScoreAttribute, 'float', this.candidateCount ).setName( options.name ? `${ options.name }ScoreCandidates` : 'LLMScoreCandidates' );
		this.greedyComputeNodes = [];
		this.computeLevels = [];

		for ( let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex ++ ) {

			this.greedyComputeNodes.push( this.createChunkPartialMaxNode(
				chunks[ chunkIndex ],
				chunkIndex,
				options.name ? `${ options.name }Chunk${ chunkIndex }PartialMax` : `LLMLogitChunk${ chunkIndex }PartialMax`
			) );

		}

		this.greedyComputeNodes.push( this.createGreedyMergeNode( options.name ? `${ options.name }Greedy` : 'LLMLogitGreedy' ) );

		for ( let rank = 0; rank < this.candidateCount; rank ++ ) {

			const nodes = [];

			for ( let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex ++ ) {

				nodes.push( this.createChunkCandidateNode(
					chunks[ chunkIndex ],
					chunkIndex,
					rank,
					options.name ? `${ options.name }Chunk${ chunkIndex }Candidate${ rank }` : `LLMLogitChunk${ chunkIndex }Candidate${ rank }`
				) );

			}

			nodes.push( this.createGlobalCandidateNode( rank, options.name ? `${ options.name }Candidate${ rank }` : `LLMLogitCandidate${ rank }` ) );
			this.computeLevels.push( nodes );

		}

	}

	applySoftcap( value ) {

		const cap = this.logitSoftcap;

		if ( cap === null || cap === undefined ) return value;

		return float( cap ).mul( tanh( value.div( float( cap ) ) ) );

	}

	chunkCandidateIndex( chunkIndex, rank ) {

		return chunkIndex * this.candidateCount + rank;

	}

	createChunkPartialMaxNode( chunk, chunkIndex, name ) {

		const { partialTokenNode, partialScoreNode, partialOffsets, workgroupSize } = this;
		const localTokens = workgroupArray( 'uint', workgroupSize );
		const localScores = workgroupArray( 'float', workgroupSize );

		return Fn( () => {

			const localIndex = invocationLocalIndex;
			const tokenIndex = instanceIndex;
			const globalToken = uint( chunk.offset ).add( tokenIndex );
			const valid = tokenIndex.lessThan( uint( chunk.size ) );
			const score = float( LOWEST_FLOAT ).toVar( `partialScore${ chunkIndex }` );

			If( valid, () => {

				score.assign( this.applySoftcap( chunk.layer.outputNode.element( tokenIndex ) ) );

			} );

			localTokens.element( localIndex ).assign( globalToken );
			localScores.element( localIndex ).assign( score );
			workgroupBarrier();

			for ( let stride = workgroupSize / 2; stride >= 1; stride /= 2 ) {

				If( localIndex.lessThan( uint( stride ) ), () => {

					const otherIndex = localIndex.add( uint( stride ) );
					const otherScore = localScores.element( otherIndex );
					const otherToken = localTokens.element( otherIndex );
					const bestScore = localScores.element( localIndex );
					const bestToken = localTokens.element( localIndex );
					const betterScore = otherScore.greaterThan( bestScore );
					const earlierTie = otherScore.equal( bestScore ).and( otherToken.lessThan( bestToken ) );

					If( betterScore.or( earlierTie ), () => {

						localScores.element( localIndex ).assign( otherScore );
						localTokens.element( localIndex ).assign( otherToken );

					} );

				} );

				workgroupBarrier();

			}

			If( localIndex.equal( uint( 0 ) ), () => {

				const partialIndex = uint( partialOffsets[ chunkIndex ] ).add( workgroupId.x );
				partialTokenNode.element( partialIndex ).assign( localTokens.element( uint( 0 ) ) );
				partialScoreNode.element( partialIndex ).assign( localScores.element( uint( 0 ) ) );

			} );

		} )().compute( chunk.size, [ workgroupSize ] ).setName( name );

	}

	createGreedyMergeNode( name ) {

		const { candidateTokenNode, candidateScoreNode, partialTokenNode, partialScoreNode, partialCount } = this;

		return Fn( () => {

			const bestToken = uint( 0 ).toVar( 'bestToken' );
			const bestScore = float( LOWEST_FLOAT ).toVar( 'bestScore' );

			Loop( { start: uint( 0 ), end: uint( partialCount ), type: 'uint', condition: '<' }, ( { i } ) => {

				const tokenId = partialTokenNode.element( i );
				const score = partialScoreNode.element( i );
				const betterScore = score.greaterThan( bestScore );
				const earlierTie = score.equal( bestScore ).and( tokenId.lessThan( bestToken ) );

				If( betterScore.or( earlierTie ), () => {

					bestToken.assign( tokenId );
					bestScore.assign( score );

				} );

			} );

			candidateTokenNode.element( uint( 0 ) ).assign( bestToken );
			candidateScoreNode.element( uint( 0 ) ).assign( bestScore );

		} )().compute( 1, [ 1 ] ).setName( name );

	}

	createChunkCandidateNode( chunk, chunkIndex, rank, name ) {

		const { chunkCandidateTokenNode, chunkCandidateScoreNode } = this;
		const candidateOffset = this.chunkCandidateIndex( chunkIndex, 0 );

		return Fn( () => {

			const bestToken = uint( 0 ).toVar( 'bestToken' );
			const bestScore = float( LOWEST_FLOAT ).toVar( 'bestScore' );

			Loop( { start: uint( 0 ), end: uint( chunk.size ), type: 'uint', condition: '<' }, ( { i } ) => {

				const tokenId = uint( chunk.offset ).add( i );
				const score = this.applySoftcap( chunk.layer.outputNode.element( i ) );
				const selectedFlag = uint( 0 ).toVar( `selectedFlag${ chunkIndex }_${ rank }` );

				for ( let previousRank = 0; previousRank < rank; previousRank ++ ) {

					If( chunkCandidateTokenNode.element( uint( candidateOffset + previousRank ) ).equal( tokenId ), () => {

						selectedFlag.assign( uint( 1 ) );

					} );

				}

				const betterScore = score.greaterThan( bestScore );
				const earlierTie = score.equal( bestScore ).and( tokenId.lessThan( bestToken ) );

				If( selectedFlag.equal( uint( 0 ) ).and( betterScore.or( earlierTie ) ), () => {

					bestToken.assign( tokenId );
					bestScore.assign( score );

				} );

			} );

			chunkCandidateTokenNode.element( uint( candidateOffset + rank ) ).assign( bestToken );
			chunkCandidateScoreNode.element( uint( candidateOffset + rank ) ).assign( bestScore );

		} )().compute( 1, [ 1 ] ).setName( name );

	}

	createGlobalCandidateNode( rank, name ) {

		const { candidateTokenNode, candidateScoreNode, chunkCandidateTokenNode, chunkCandidateScoreNode, chunks } = this;

		return Fn( () => {

			const bestToken = uint( 0 ).toVar( 'bestToken' );
			const bestScore = float( LOWEST_FLOAT ).toVar( 'bestScore' );

			for ( let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex ++ ) {

				for ( let localRank = 0; localRank <= rank; localRank ++ ) {

					const candidateIndex = uint( this.chunkCandidateIndex( chunkIndex, localRank ) );
					const tokenId = chunkCandidateTokenNode.element( candidateIndex );
					const score = chunkCandidateScoreNode.element( candidateIndex );
					const selectedFlag = uint( 0 ).toVar( `globalSelectedFlag${ rank }_${ chunkIndex }_${ localRank }` );

					for ( let previousRank = 0; previousRank < rank; previousRank ++ ) {

						If( candidateTokenNode.element( uint( previousRank ) ).equal( tokenId ), () => {

							selectedFlag.assign( uint( 1 ) );

						} );

					}

					const betterScore = score.greaterThan( bestScore );
					const earlierTie = score.equal( bestScore ).and( tokenId.lessThan( bestToken ) );

					If( selectedFlag.equal( uint( 0 ) ).and( betterScore.or( earlierTie ) ), () => {

						bestToken.assign( tokenId );
						bestScore.assign( score );

					} );

				}

			}

			candidateTokenNode.element( uint( rank ) ).assign( bestToken );
			candidateScoreNode.element( uint( rank ) ).assign( bestScore );

		} )().compute( 1, [ 1 ] ).setName( name );

	}

	computeNodesFor( count ) {

		if ( count <= 1 ) return this.greedyComputeNodes;

		const nodes = [];
		const candidateCount = Math.min( Math.max( 1, count ), this.candidateCount );

		for ( let i = 0; i < candidateCount; i ++ ) nodes.push( ...this.computeLevels[ i ] );

		return nodes;

	}

	async readToken( renderer ) {

		return new Uint32Array( await renderer.getArrayBufferAsync( this.candidateTokenAttribute, null, 0, 4 ) )[ 0 ];

	}

	async readCandidates( renderer, count ) {

		const candidateCount = Math.min( Math.max( 1, count ), this.candidateCount );
		const byteCount = candidateCount * 4;
		const tokens = new Uint32Array( await renderer.getArrayBufferAsync( this.candidateTokenAttribute, null, 0, byteCount ) );
		const scores = new Float32Array( await renderer.getArrayBufferAsync( this.candidateScoreAttribute, null, 0, byteCount ) );
		const candidates = [];

		for ( let i = 0; i < candidateCount; i ++ ) candidates.push( [ tokens[ i ], scores[ i ] ] );

		return candidates;

	}

	async sampleToken( renderer, count, options ) {

		if ( count <= 1 || options.temperature <= 0 ) return this.readToken( renderer );

		return sampleTopKCandidates( await this.readCandidates( renderer, count ), options );

	}

}

function createLogitSampler( chunks, options ) {

	return new TSLLogitSampler( chunks, options );

}

export { TSLLogitSampler, createChunkedLogitLayers, createLogitSampler, readChunkedLogits };

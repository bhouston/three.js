import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, instanceIndex, storage, uint } from 'three/tsl';

/**
 * Concatenate several 1D storage vectors into one buffer.
 *
 * @three_import import { TSLConcat } from 'three/addons/gpgpu/llm/TSLConcat.js';
 */
class TSLConcat {

	constructor( parts, options = {} ) {

		this.parts = parts;
		this.size = parts.reduce( ( sum, part ) => sum + part.size, 0 );
		this.outputAttribute = new StorageBufferAttribute( new Float32Array( this.size ), 1 );
		this.outputNode = storage( this.outputAttribute, 'float', this.size ).setName( options.name ? `${ options.name }Output` : 'LLMConcatOutput' );
		this.computeNode = this.createComputeNode( options.name || 'LLMConcat', options.workgroupSize || 64 );

	}

	createComputeNode( name, workgroupSize ) {

		const { parts, outputNode, size } = this;
		const ranges = [];
		let offset = 0;

		for ( let i = 0; i < parts.length; i ++ ) {

			ranges.push( { node: parts[ i ].node, start: offset, end: offset + parts[ i ].size } );
			offset += parts[ i ].size;

		}

		return Fn( () => {

			const index = instanceIndex.toVar( 'index' );

			If( index.lessThan( uint( size ) ), () => {

				for ( let i = 0; i < ranges.length; i ++ ) {

					const range = ranges[ i ];
					If( index.greaterThanEqual( uint( range.start ) ).and( index.lessThan( uint( range.end ) ) ), () => {

						outputNode.element( index ).assign( range.node.element( index.sub( uint( range.start ) ) ) );

					} );

				}

			} );

		} )().compute( size, [ workgroupSize ] ).setName( name );

	}

	compute( renderer ) {

		renderer.compute( this.computeNode );
		return this.outputNode;

	}

}

export { TSLConcat };

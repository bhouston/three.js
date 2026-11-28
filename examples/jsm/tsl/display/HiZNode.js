import { Node, NodeUpdateType, StorageBufferAttribute, Vector2, Vector4 } from 'three/webgpu';
import { Fn, If, float, vec2, int, uint, uvec2, ivec2, max, min, nodeObject, storage, textureLoad, uniform, uniformArray, instanceIndex, uv } from 'three/tsl';

// the maximum number of levels, enough for 65536 pixels

const MAX_LEVELS = 16;

/**
 * A hierarchical depth buffer, built from a depth texture by one compute dispatch per level.
 *
 * Each level stores the farthest depth of 2x2 texels of the previous level. The depth texture is not copied, the
 * first level covers 2x2 pixels, so a texel of a level covers `2^( level + 1 )` pixels. The sizes are rounded up
 * and the texels at the border cover the remaining pixels, so odd sizes stay conservative. The levels are stored
 * in one storage buffer.
 *
 * With `nearest` each texel is a `vec2` of the nearest and the farthest depth, otherwise a `float` of the
 * farthest depth. The depths are raw device depths.
 *
 * The hierarchy is updated before each frame when the node is part of a rendered node graph. Otherwise dispatch
 * {@link HiZNode#computeNodes} after the depth has been rendered.
 *
 * ```js
 * const hiZNode = hiZ( scenePass.getTextureNode( 'depth' ) );
 * hiZNode.level.value = 3;
 * renderPipeline.outputNode = vec4( vec3( hiZNode ), 1 );
 * ```
 *
 * @augments Node
 * @three_import import { hiZ } from 'three/addons/tsl/display/HiZNode.js';
 */
class HiZNode extends Node {

	static get type() {

		return 'HiZNode';

	}

	/**
	 * Constructs a new Hi-Z node.
	 *
	 * @param {TextureNode} depthNode - The raw depth.
	 * @param {Object} [options] - The options.
	 * @param {boolean} [options.nearest=false] - Whether the nearest depth is stored next to the farthest depth.
	 * @param {boolean} [options.reversedDepth] - Whether the depth is reversed, the depth of the renderer by default.
	 */
	constructor( depthNode, { nearest = false, reversedDepth } = {} ) {

		super( nearest ? 'vec2' : 'float' );

		this.depthNode = depthNode;
		this.nearest = nearest;
		this.reversedDepth = reversedDepth;

		/**
		 * The level returned by the node itself, for debugging.
		 *
		 * @type {UniformNode<uint>}
		 */
		this.level = uniform( 0, 'uint' );

		/**
		 * The size of the depth in pixels.
		 *
		 * @type {UniformNode<vec2>}
		 */
		this.size = uniform( new Vector2() );

		/**
		 * The number of levels.
		 *
		 * @type {UniformNode<uint>}
		 */
		this.levelCount = uniform( 0, 'uint' );

		/**
		 * The compute nodes of the levels, dispatched in order.
		 *
		 * @type {Array<ComputeNode>}
		 */
		this.computeNodes = [];

		this.updateBeforeType = NodeUpdateType.FRAME;

		this._levels = [];
		this._levelsNode = uniformArray( Array.from( { length: MAX_LEVELS }, () => new Vector4() ), 'vec4' );
		this._attribute = null;
		this._write = null;
		this._read = null;
		this._reversed = reversedDepth === true;

		const image = depthNode.value.image;

		this.setSize( image.width || 1, image.height || 1 );

	}

	/**
	 * Returns the size of a level and a function loading its texels.
	 *
	 * @param {Node<int>} level - The level, it must be less than {@link HiZNode#levelCount}.
	 * @return {{width: Node<uint>, height: Node<uint>, load: Function}} `load( x, y )` returns the texel.
	 */
	getLevelNode( level ) {

		const data = this._levelsNode.element( level );
		const offset = uint( data.x );
		const width = uint( data.y );
		const height = uint( data.z );

		return { width, height, load: ( x, y ) => this._read.element( offset.add( y.mul( width ) ).add( x ) ) };

	}

	/**
	 * Returns the texel of a level covering the given uv.
	 *
	 * @param {Node<uint>} [level=this.level] - The level, clamped to the last level.
	 * @param {Node<vec2>} [uvNode=uv()] - The uv.
	 * @return {Node<float|vec2>} The farthest depth, or the nearest and the farthest depth.
	 */
	getTexelNode( level = this.level, uvNode = uv() ) {

		const clamped = min( uint( level ), this.levelCount.sub( 1 ) );
		const { width, height, load } = this.getLevelNode( int( clamped ) );
		const cell = min( uvec2( uvNode.mul( this.size ) ).shiftRight( uvec2( clamped.add( 1 ) ) ), uvec2( width, height ).sub( 1 ) );

		return load( cell.x, cell.y );

	}

	/**
	 * Returns the cell of a mip covering a pixel, for ray traversals such as `traceHiZ()`. Mip `0` is the depth
	 * itself and mip `m` is the level `m - 1`, so the last mip is {@link HiZNode#levelCount}. Requires `nearest`.
	 *
	 * @param {Node<uint>} mip - The mip, it must not exceed {@link HiZNode#levelCount}.
	 * @param {Node<vec2>} pixelNode - The pixel coordinates in the depth.
	 * @return {{cell: Node<uvec2>, depth: Node<float>, farthest: Node<float>, begin: Node<vec2>, end: Node<vec2>}} The cell, its nearest and farthest depth and its pixel rectangle.
	 */
	getPixelCellNode( mip, pixelNode ) {

		if ( this.nearest === false ) throw new Error( 'THREE.HiZNode: getPixelCellNode() requires the nearest depth.' );

		mip = uint( mip );

		const shift = uvec2( mip );
		const size = uvec2( this.size ).toConst();
		const cell = min( uvec2( max( pixelNode, 0 ) ), size.sub( 1 ) ).shiftRight( shift ).toConst();
		const begin = cell.shiftLeft( shift ).toConst();
		const end = min( begin.add( uvec2( 1 ).shiftLeft( shift ) ), size );
		const texel = vec2( 0 ).toVar();

		If( mip.equal( 0 ), () => {

			texel.assign( textureLoad( this.depthNode, ivec2( cell ) ).xx );

		} ).Else( () => {

			texel.assign( this.getLevelNode( int( mip ).sub( 1 ) ).load( cell.x, cell.y ) );

		} );

		return { cell, depth: texel.x, farthest: texel.y, begin: vec2( begin ), end: vec2( end ) };

	}

	/**
	 * Sets the size of the depth, called automatically when the node updates itself.
	 *
	 * @param {number} width - The width in pixels.
	 * @param {number} height - The height in pixels.
	 */
	setSize( width, height ) {

		if ( this.size.value.x === width && this.size.value.y === height ) return;

		this.size.value.set( width, height );

		const levels = this._levels = [];

		let size = 0;

		do {

			width = Math.ceil( width / 2 );
			height = Math.ceil( height / 2 );

			levels.push( { offset: size, width, height } );

			size += width * height;

		} while ( width > 1 || height > 1 );

		this._levelsNode.array.forEach( ( data, i ) => {

			const level = levels[ Math.min( i, levels.length - 1 ) ];

			data.set( level.offset, level.width, level.height, 0 );

		} );

		this.levelCount.value = levels.length;

		if ( this._attribute !== null ) this._attribute.dispose();

		const itemSize = this.nearest ? 2 : 1;
		const type = this.nearest ? 'vec2' : 'float';

		this._attribute = new StorageBufferAttribute( new Float32Array( size * itemSize ), itemSize );

		if ( this._write === null ) {

			this._write = storage( this._attribute, type );
			this._read = storage( this._attribute, type ).toReadOnly();

		} else {

			this._write.value = this._read.value = this._attribute;

		}

		this._build();

	}

	_build() {

		for ( const computeNode of this.computeNodes ) computeNode.dispose();

		const { nearest, depthNode } = this;
		const levels = this._levels;
		const buffer = this._write;

		// the farthest depth is the maximum, the minimum with a reversed depth

		const far = this._reversed ? min : max;
		const near = this._reversed ? max : min;
		const farInitial = this._reversed ? 1 : 0;

		this.computeNodes = levels.map( ( level, i ) => {

			const source = i > 0 ? levels[ i - 1 ] : { offset: 0, width: this.size.value.x, height: this.size.value.y };

			return Fn( () => {

				If( instanceIndex.lessThan( level.width * level.height ), () => {

					const x = instanceIndex.mod( level.width );
					const y = instanceIndex.div( level.width );

					const depth = ( nearest ? vec2( 1 - farInitial, farInitial ) : float( farInitial ) ).toVar();

					for ( let j = 0; j < 4; j ++ ) {

						const sourceX = min( x.mul( 2 ).add( j & 1 ), uint( source.width - 1 ) );
						const sourceY = min( y.mul( 2 ).add( j >> 1 ), uint( source.height - 1 ) );

						const value = i === 0 ? textureLoad( depthNode, ivec2( sourceX, sourceY ) ).x : buffer.element( sourceY.mul( source.width ).add( sourceX ).add( source.offset ) );

						if ( nearest ) {

							depth.assign( i === 0 ? vec2( near( depth.x, value ), far( depth.y, value ) ) : vec2( near( depth.x, value.x ), far( depth.y, value.y ) ) );

						} else {

							depth.assign( far( depth, value ) );

						}

					}

					buffer.element( instanceIndex.add( level.offset ) ).assign( depth );

				} );

			} )().compute( level.width * level.height ).setName( 'HiZNode Level ' + i );

		} );

	}

	updateBefore( { renderer } ) {

		const image = this.depthNode.value.image;

		this.setSize( image.width, image.height );

		const reversed = this.reversedDepth ?? renderer.reversedDepthBuffer === true;

		if ( reversed !== this._reversed ) {

			this._reversed = reversed;
			this._build();

		}

		renderer.compute( this.computeNodes );

	}

	setup( builder ) {

		// the pass rendering the depth has to update before this node

		builder.getNodeProperties( this ).depthNode = this.depthNode;

		return this.getTexelNode();

	}

	dispose() {

		super.dispose();

		for ( const computeNode of this.computeNodes ) computeNode.dispose();

		this._attribute.dispose();

	}

}

export default HiZNode;

/**
 * TSL function for creating a hierarchical depth buffer.
 *
 * @tsl
 * @function
 * @param {TextureNode} depthNode - The raw depth.
 * @param {Object} [options] - The options, see {@link HiZNode}.
 * @return {HiZNode} The Hi-Z node.
 */
export const hiZ = ( depthNode, options ) => new HiZNode( nodeObject( depthNode ), options );

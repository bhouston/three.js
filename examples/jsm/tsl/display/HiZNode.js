import { FloatType, NearestFilter, Node, NodeUpdateType, RedFormat, StorageTexture, Vector4 } from 'three/webgpu';
import { Fn, If, Loop, float, invocationLocalIndex, ivec2, max, min, nodeObject, passTexture, textureLoad, textureStore, uint, uniform, uniformArray, uv, vec4, workgroupArray, workgroupBarrier, workgroupId } from 'three/tsl';

const WORKGROUP_SIZE = 64;
const MAX_LEVELS = 16;

/**
 * Builds a farthest-depth hierarchy in one WebGPU compute dispatch.
 *
 * Levels are stored vertically in an R32F atlas, with floor-halved dimensions.
 * The last cell includes the odd trailing row/column (a 3x2, 2x3 or 3x3
 * reduction). Standard depth uses max; reversed depth uses min. Values remain
 * raw device depth, not linearized depth.
 *
 * Each level reduces the original depth independently. This avoids relying on
 * cross-workgroup publication, which WGSL does not guarantee. Coarse cells use
 * cooperative workgroup reductions, but source reads still scale with the
 * number of levels. This is not an SPD implementation or an SSR traversal.
 *
 * ```js
 * const pyramid = hiZ( scenePass.getTextureNode( 'depth' ) );
 * pyramid.level.value = 3;
 * renderPipeline.outputNode = vec4( vec3( pyramid ), 1 );
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
	 * @param {TextureNode} depthNode - A full-resolution, raw depth texture node.
	 * @param {Object} [options] - Options.
	 * @param {boolean} [options.reversedDepth] - Override the renderer's depth convention (useful for synthetic depth textures).
	 */
	constructor( depthNode, { reversedDepth } = {} ) {

		super( 'float' );

		this.depthNode = depthNode;
		this.reversedDepth = reversedDepth;
		this.enabled = true;
		this.level = uniform( 0, 'uint' );
		this.updateBeforeType = NodeUpdateType.FRAME;

		this.texture = new StorageTexture();
		this.texture.name = 'HiZNode.Atlas';
		this.texture.type = FloatType;
		this.texture.format = RedFormat;
		this.texture.minFilter = NearestFilter;
		this.texture.magFilter = NearestFilter;
		this.texture.generateMipmaps = false;

		/** @type {Array<{width:number,height:number,offset:number}>} */
		this.levels = [];
		this._layout = uniformArray( Array.from( { length: MAX_LEVELS }, () => new Vector4( 1, 1, 0, 0 ) ) );
		this._lastLevel = uniform( 0, 'uint' );
		this._textureNode = passTexture( this, this.texture );
		this._computeNode = null;
		this._width = 0;
		this._height = 0;
		this._reversed = null;
		this._dispatch = [ 1, 1, 1 ];

	}

	/**
	 * Samples a hierarchy level using nearest texels, clamped within that level.
	 * Level zero is the original depth. Out-of-range levels clamp to the last.
	 *
	 * @param {Node<uint>|number} [level=this.level] - Hierarchy level.
	 * @param {Node<vec2>} [uvNode=uv()] - Normalized coordinates.
	 * @return {Node<float>} Raw farthest depth.
	 */
	getTextureNode( level = this.level, uvNode = uv() ) {

		const layout = this._layout.element( uint( level ).min( this._lastLevel ) );
		const coord = ivec2( uvNode.mul( layout.xy ).floor().clamp( 0, layout.xy.sub( 1 ) ) );
		return textureLoad( this._textureNode, coord.add( ivec2( 0, layout.z ) ) ).r;

	}

	/**
	 * Updates the hierarchy layout. Called automatically from the input size.
	 *
	 * @param {number} width - Input width in pixels.
	 * @param {number} height - Input height in pixels.
	 */
	setSize( width, height ) {

		if ( ! Number.isInteger( width ) || ! Number.isInteger( height ) || width < 1 || height < 1 ) {

			throw new Error( 'HiZNode: Depth dimensions must be positive integers.' );

		}

		if ( width === this._width && height === this._height ) return;

		this._width = width;
		this._height = height;
		this.levels = [];
		let offset = 0;

		while ( true ) {

			this.levels.push( { width, height, offset } );
			offset += height;
			if ( width === 1 && height === 1 ) break;
			width = Math.max( 1, Math.floor( width / 2 ) );
			height = Math.max( 1, Math.floor( height / 2 ) );

		}

		if ( this.levels.length > MAX_LEVELS ) throw new Error( 'HiZNode: Too many hierarchy levels.' );

		for ( let i = 0; i < MAX_LEVELS; i ++ ) {

			const entry = this.levels[ Math.min( i, this.levels.length - 1 ) ];
			this._layout.array[ i ].set( entry.width, entry.height, entry.offset, 0 );

		}

		this._lastLevel.value = this.levels.length - 1;
		this.texture.setSize( this._width, offset );
		this._disposeCompute();

	}

	/** @private */
	_disposeCompute() {

		if ( this._computeNode !== null ) this._computeNode.dispose();
		this._computeNode = null;

	}

	/** @private */
	_buildCompute() {

		const width = this._width;
		const height = this._height;
		const reversed = this._reversed;
		const reduce = reversed ? min : max;
		const identity = reversed ? 1 : 0;
		const shared = workgroupArray( 'float', WORKGROUP_SIZE );
		let groupCount = 0;
		const tasks = this.levels.map( ( entry, level ) => {

			// Small footprints use one lane per texel; large footprints use a
			// whole workgroup so the final 1x1 level never becomes a serial scan.
			const cooperative = level >= 4;
			const count = cooperative ? entry.width * entry.height : Math.ceil( entry.width * entry.height / WORKGROUP_SIZE );
			const task = { ...entry, level, cooperative, start: groupCount, count };
			groupCount += count;
			return task;

		} );

		const dispatchX = Math.min( groupCount, 65535 );
		this._dispatch = [ dispatchX, Math.ceil( groupCount / dispatchX ), 1 ];

		this._computeNode = Fn( () => {

			const group = workgroupId.x.add( workgroupId.y.mul( dispatchX ) ).toVar();
			const lane = invocationLocalIndex;

			for ( const task of tasks ) {

				const { level, cooperative, start, count, offset } = task;
				const scale = 2 ** level;

				If( group.greaterThanEqual( start ).and( group.lessThan( start + count ) ), () => {

					const index = cooperative ? group.sub( start ) : group.sub( start ).mul( WORKGROUP_SIZE ).add( lane );
					const x = index.mod( task.width ).toVar();
					const y = index.div( task.width ).toVar();
					const beginX = x.mul( scale ).toVar();
					const beginY = y.mul( scale ).toVar();
					const endX = x.equal( task.width - 1 ).select( uint( width ), beginX.add( scale ) );
					const endY = y.equal( task.height - 1 ).select( uint( height ), beginY.add( scale ) );
					const spanX = endX.sub( beginX ).toVar();
					const samples = spanX.mul( endY.sub( beginY ) ).toVar();
					const value = float( identity ).toVar();

					const read = ( i ) => {

						const coord = ivec2( beginX.add( i.mod( spanX ) ), beginY.add( i.div( spanX ) ) );
						value.assign( reduce( value, textureLoad( this.depthNode, coord ).r ) );

					};

					if ( cooperative ) {

						Loop( { start: lane, end: samples, type: 'uint', update: `+= ${ WORKGROUP_SIZE }` }, ( { i } ) => read( i ) );
						shared.element( lane ).assign( value );
						workgroupBarrier();

						for ( let stride = WORKGROUP_SIZE / 2; stride >= 1; stride /= 2 ) {

							If( lane.lessThan( stride ), () => {

								shared.element( lane ).assign( reduce( shared.element( lane ), shared.element( lane.add( stride ) ) ) );

							} );
							workgroupBarrier();

						}

						If( lane.equal( 0 ), () => {

							textureStore( this.texture, ivec2( x, y.add( offset ) ), vec4( shared.element( 0 ) ) ).toWriteOnly();

						} );

					} else {

						If( index.lessThan( task.width * task.height ), () => {

							Loop( { start: uint( 0 ), end: samples, type: 'uint' }, ( { i } ) => read( i ) );
							textureStore( this.texture, ivec2( x, y.add( offset ) ), vec4( value ) ).toWriteOnly();

						} );

					}

				} );

			}

		} )().compute( this._dispatch, [ WORKGROUP_SIZE ] ).setName( 'Hi-Z [ Pyramid ]' );

	}

	/**
	 * @param {NodeFrame} frame - Frame state.
	 */
	updateBefore( { renderer } ) {

		if ( this.enabled === false ) return;
		if ( renderer.backend.isWebGPUBackend !== true ) throw new Error( 'HiZNode: A WebGPU backend is required.' );

		const image = this.depthNode.value.image;
		this.setSize( image.width, image.height );
		const limit = renderer.backend.device.limits.maxTextureDimension2D;
		if ( this.texture.image.width > limit || this.texture.image.height > limit ) throw new Error( 'HiZNode: Depth atlas exceeds maxTextureDimension2D.' );

		const reversed = this.reversedDepth ?? renderer.reversedDepthBuffer;
		if ( reversed !== this._reversed ) {

			this._reversed = reversed;
			this._disposeCompute();

		}

		if ( this._computeNode === null ) this._buildCompute();
		renderer.compute( this._computeNode );

	}

	/**
	 * @param {NodeBuilder} builder - Node builder.
	 * @return {Node<float>} Selected level.
	 */
	setup( builder ) {

		// Make the producing pass a dependency of this node, before dispatch.
		builder.getNodeProperties( this ).depthNode = this.depthNode;
		return this.getTextureNode();

	}

	/** Frees the atlas and compute pipeline. */
	dispose() {

		super.dispose();
		this._disposeCompute();
		this.texture.dispose();

	}

}

export default HiZNode;

/**
 * Creates a farthest-depth pyramid. Requires a hardware WebGPU backend.
 *
 * @tsl
 * @function
 * @param {TextureNode} depthNode - Raw depth texture.
 * @param {Object} [options] - Depth convention override.
 * @return {HiZNode} The depth hierarchy node.
 */
export const hiZ = ( depthNode, options ) => new HiZNode( nodeObject( depthNode ), options );

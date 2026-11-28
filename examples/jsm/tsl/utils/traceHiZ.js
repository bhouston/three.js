import { Break, Continue, Fn, If, Loop, bool, float, int, max, min, projectionDepth, uint, uniform, uvec2, vec2 } from 'three/tsl';
import { hiZ } from '../display/HiZNode.js';

// selects by the direction of the depth buffer when the shader is built

const ifReversed = /*@__PURE__*/ Fn( ( [ reversed, standard ], builder ) => builder.renderer.reversedDepthBuffer === true ? reversed : standard );

/**
 * Builds hierarchical screen-space traversal inside the calling TSL function.
 * UV and projected device depth are affine in the same ray parameter, [0,1].
 * The endpoints must already be clipped to the viewport and camera near plane.
 *
 * Cells are skipped only while the ray is in front of their nearest surface.
 * Potential intersections descend to mip zero. Both depth directions are solved,
 * including a ray approaching a surface from behind while moving toward the camera.
 * The ray stops at the first surface it reaches. A caller-supplied predicate then
 * performs full-resolution normal/thickness validation once, outside the loop.
 *
 * Derived from the adaptive traversal in three.js PR #35228. Kept separate from
 * ray generation and shading, so it is shared by the effects tracing the depth.
 *
 * @param {HiZNode} hierarchy - Hierarchical depth of the scene being traced, with the nearest depth.
 * @param {Node<vec3>} origin - Start UV and projected depth.
 * @param {Node<vec3>} end - End UV and projected depth.
 * @param {Object} options - Traversal configuration.
 * @param {Node<float>} options.near - The camera's near value.
 * @param {Node<float>} options.far - The camera's far value.
 * @param {Node<float>|number} [options.start=0] - Initial ray parameter (self-hit exclusion).
 * @param {Node<int>|number} [options.maxSteps=32] - Work budget.
 * @param {Node<float>|number} [options.depthTolerance=0] - Conservative projected-depth rounding tolerance.
 * @param {Node<int>|number|Function} [options.minLevel=0] - Finest level later descents need to reach (FidelityFX's most_detailed_mip), or a function of the level-zero pixels travelled so far returning it, for a tolerance that grows along the ray. A coarse leaf is accepted only when the full-resolution pixel under the ray is itself the hit; otherwise that cell is resolved exactly, so hits match `minLevel = 0`.
 * @param {boolean} [options.coarseHits=false] - Cone-tracing approximation: a coarse leaf is tested once, with its nearest depth at the ray's UV, and stepped over when `acceptHit` rejects that. `acceptHit` receives the cell size in pixels as a fourth argument to widen its tolerance.
 * @param {Function} [options.isHidden] - TSL predicate (entry UV, exit UV, raw farthest depth): true when no surface in the cell can be accepted because the ray is too far behind all of them. Such cells are skipped like empty ones.
 * @param {Function} [options.acceptHit] - TSL predicate accepting (UV, parameter, raw depth).
 * @param {boolean} [options.anyHit=false] - Whether any accepted surface on the ray is returned instead of the closest one, for visibility rays. A coarse cell is then hit without descending, at a point where the ray is behind all of its surfaces and `acceptHit` holds for both its nearest and its farthest depth. `acceptHit` must hold for every depth between two depths it accepts. The depth of the hit is the nearest depth of the cell.
 * @param {boolean} [options.passRejected=false] - Whether the ray continues past a surface `acceptHit` refuses, instead of stopping there. For visibility rays, which only need any accepted occluder.
 * @return {Object} Hit flag, UV, raw depth, parameter, iterations, exhausted flag and rejected flag: the ray reached a
 * surface that `acceptHit` refused, so it is blocked rather than free.
 */
export function traceHiZ( hierarchy, origin, end, { near, far, start = 0, maxSteps = 32, depthTolerance = 0, minLevel = 0, coarseHits = false, isHidden = null, acceptHit = () => bool( true ), anyHit = false, passRejected = false } = {} ) {

	const delta = end.sub( origin ).toConst();
	const s = float( start ).toVar();
	// Traversal always starts at mip zero: the cells next to the origin hold its
	// own surface, which only full resolution separates from the ray.
	const mip = int( 0 ).toVar();
	const coarse = minLevel !== 0;
	const minMip = coarse ? int( 0 ).toVar() : int( 0 );
	// Set when a coarse leaf failed validation: descend to mip zero until the
	// traversal climbs back to the wanted level.
	const exact = bool( false ).toVar();
	const validated = bool( false ).toVar();
	const lastMip = int( hierarchy.levelCount ).toConst();
	const hit = bool( false ).toVar();
	const hitUV = vec2( 0 ).toVar();
	const hitDepth = float( 0 ).toVar();
	const hitS = float( 0 ).toVar();
	const iterations = int( 0 ).toVar();
	const size = hierarchy.size;
	// Traversal runs in level-zero pixels: cells are integer shifts of them.
	const originPixel = origin.xy.mul( size ).toConst();
	const deltaPixel = delta.xy.mul( size ).toConst();
	const pixels = deltaPixel.abs();
	const rayPixels = deltaPixel.length().toConst();
	const exitParity = delta.xy.greaterThanEqual( vec2( 0 ) ).select( uvec2( 1 ), uvec2( 0 ) ).toConst();
	// Ray-space progress can be smaller than one pixel ULP on the nearly
	// stationary axis. Every lookup is taken a fraction of a pixel ahead of the
	// ray, so a point on a cell edge resolves to the cell being entered at every
	// level alike and coarse/fine mips cannot oscillate at the same wall.
	const edgeBias = size.mul( 2 ** - 22 ).max( 0.0005 );
	const lookupOrigin = originPixel.add( delta.xy.sign().mul( edgeBias ) ).toConst();
	// Offset only XY exits, never depth-plane intersections. The offset is a
	// hundred-thousandth of a source texel, with a float32 progress floor in ray space.
	const epsilon = float( 0.00001 ).div( max( pixels.x, pixels.y ).max( 1 ) ).max( 1e-7 ).toConst();
	const inverseXY = delta.xy.greaterThanEqual( vec2( 0 ) ).select( vec2( 1 ), vec2( - 1 ) ).div( pixels.max( 1e-20 ) ).toConst();
	const inverseZ = delta.z.greaterThanEqual( 0 ).select( 1, - 1 ).div( delta.z.abs().max( 1e-20 ) ).toConst();
	const farther = ( a, b ) => ifReversed( a.lessThan( b ), a.greaterThan( b ) );
	const nearer = ( a, b ) => ifReversed( a.greaterThan( b ), a.lessThan( b ) );
	const projectDepth = ( raw ) => projectionDepth( raw, near, far );
	// the rounding tolerance of a depth, towards the camera
	const tolerance = ifReversed( float( depthTolerance ), float( depthTolerance ).negate() );

	Loop( { start: int( 0 ), end: int( maxSteps ).clamp( 1, 4096 ), type: 'int', condition: '<' }, ( { i } ) => {

		If( s.greaterThanEqual( 1 ), () => {

			Break();

		} );
		iterations.assign( i.add( 1 ) );
		if ( coarse ) {

			const wanted = int( typeof minLevel === 'function' ? minLevel( rayPixels.mul( s.sub( start ) ) ) : minLevel ).toVar();
			If( mip.greaterThanEqual( wanted ), () => {

				exact.assign( false );

			} );
			minMip.assign( exact.select( int( 0 ), wanted ) );

		}

		const pixel = lookupOrigin.add( deltaPixel.mul( s ) ).toVar();
		const z = origin.z.add( delta.z.mul( s ) ).toVar();
		If( pixel.lessThan( vec2( 0 ) ).any().or( pixel.greaterThanEqual( size ).any() ), () => {

			s.assign( 1 );
			Break();

		} );

		const { cell, depth: cellDepth, farthest: cellFarthest, begin, end: cellEnd } = hierarchy.getPixelCellNode( uint( mip ), pixel );
		// Zero XY components use a large positive exit distance: the origin
		// remains inside that axis's half-open cell. No per-axis branch needed.
		const edge = delta.xy.greaterThanEqual( vec2( 0 ) ).select( cellEnd, begin );
		const exits = edge.sub( originPixel ).mul( inverseXY );
		const exit = min( min( exits.x, exits.y ), 1 ).toVar();

		const leaveCell = () => {

			s.assign( max( exit.add( epsilon ), s.add( epsilon ) ) );
			// Ascend only when leaving the parent cell. Inside the same parent
			// the coarser test would repeat and cost a second read to descend.
			const leavesParent = cell.bitAnd( uvec2( 1 ) ).equal( exitParity );
			If( exits.x.lessThanEqual( exits.y ).select( leavesParent.x, leavesParent.y ), () => {

				mip.assign( mip.add( 1 ).min( lastMip ) );

			} );
			Continue();

		};

		const raw = cellDepth.toVar();
		const depth = projectDepth( raw ).toVar();
		const plane = depth.sub( origin.z ).mul( inverseZ ).toVar();
		const conservativeDepth = depth.add( tolerance );
		const inFront = nearer( z, conservativeDepth ).and(
			farther( end.z, origin.z ).not().or( plane.greaterThan( s.add( 1e-7 ) ) )
		).toVar();
		const background = ifReversed( raw.lessThanEqual( 0 ), raw.greaterThanEqual( 1 ) );

		// Behind every surface in the cell from entry to exit: the ray is hidden
		// by an occluder here and cannot be accepted, so skip it as if empty.
		const hidden = bool( false ).toVar();
		if ( isHidden !== null ) {

			const farRaw = cellFarthest.toVar();
			const farBound = projectDepth( farRaw ).sub( tolerance );
			If( inFront.not().and( background.not() ).and( farther( z, farBound ) ).and( farther( origin.z.add( delta.z.mul( exit ) ), farBound ) ), () => {

				hidden.assign( isHidden( pixel.div( size ), origin.xy.add( delta.xy.mul( exit ) ), farRaw ) );

			} );

		}

		if ( anyHit ) {

			// The point of the ray in the cell nearest to the plane of its farthest depth. Behind that plane the
			// pixel under the ray is a surface the ray has passed, whatever its depth within the cell.
			const farRaw = cellFarthest.toVar();
			const farDepth = projectDepth( farRaw ).toVar();
			const sAny = farDepth.sub( tolerance ).sub( origin.z ).mul( inverseZ ).min( exit ).max( s ).toVar();
			const zAny = origin.z.add( delta.z.mul( sAny ) );
			If( mip.greaterThan( 0 ).and( background.not() ).and( nearer( zAny, farDepth.add( tolerance ) ).not() ), () => {

				const uvAny = lookupOrigin.add( deltaPixel.mul( sAny ) ).max( begin ).min( cellEnd.sub( 0.01 ) ).div( size ).toVar();
				If( acceptHit( uvAny, sAny, raw ).and( acceptHit( uvAny, sAny, farRaw ) ), () => {

					hit.assign( true );
					validated.assign( true );
					hitUV.assign( uvAny );
					hitDepth.assign( raw );
					hitS.assign( sAny );
					Break();

				} );

			} );

		}

		If( inFront.or( background ).or( hidden ), () => {

			// An away-going ray may enter geometry before leaving this cell.
			If( background.not().and( farther( end.z, origin.z ) ).and( plane.greaterThanEqual( s ) ).and( plane.lessThan( exit ) ), () => {

				s.assign( plane );
				If( mip.greaterThan( minMip ), () => {

					mip.subAssign( 1 );
					Continue();

				} );
				// At a leaf we already have the plane and depth. Validate
				// it in this iteration, without re-reading the same leaf.
				pixel.assign( lookupOrigin.add( deltaPixel.mul( s ) ) );
				z.assign( origin.z.add( delta.z.mul( s ) ) );

			} ).Else( () => {

				leaveCell();

			} );

		} );

		If( mip.greaterThan( minMip ), () => {

			mip.subAssign( 1 );
			Continue();

		} );

		if ( coarse ) {

			if ( coarseHits ) If( mip.greaterThan( 0 ), () => {

				// Validated against the cell's nearest depth: no full-resolution depth read.
				const leafUV = pixel.div( size ).toVar();
				If( acceptHit( leafUV, s, raw, float( int( 1 ).shiftLeft( mip ) ) ), () => {

					hit.assign( true );
					validated.assign( true );
					hitUV.assign( leafUV );
					hitDepth.assign( raw );
					hitS.assign( s );
					Break();

				} );
				leaveCell();

			} );
			else If( mip.greaterThan( 0 ), () => {

				const leafUV = pixel.div( size ).toVar();
				// A coarse leaf holds its cell's nearest depth; test the full-resolution depth under the ray.
				const leaf = hierarchy.getPixelCellNode( 0, pixel ).depth.toVar();
				// The same test a mip-zero leaf makes at this pixel, so the result is
				// the exact one: only the descents in between are saved.
				const leafProjected = projectDepth( leaf ).add( tolerance );
				If( nearer( z, leafProjected ).not().and( acceptHit( leafUV, s, leaf ) ), () => {

					hit.assign( true );
					validated.assign( true );
					hitUV.assign( leafUV );
					hitDepth.assign( leaf );
					hitS.assign( s );
					Break();

				} );
				exact.assign( true );
				mip.subAssign( 1 );
				Continue();

			} );

		}

		if ( passRejected ) {

			const leafUV = pixel.div( size ).toVar();
			If( acceptHit( leafUV, s, raw ), () => {

				hit.assign( true );
				validated.assign( true );
				hitUV.assign( leafUV );
				hitDepth.assign( raw );
				hitS.assign( s );
				Break();

			} );
			leaveCell();

		}

		// AMD-style termination: return the first surface the ray reaches, then
		// validate it once outside the traversal loop.
		hit.assign( true );
		hitUV.assign( pixel.div( size ) );
		hitDepth.assign( raw );
		hitS.assign( s );
		Break();

	} );

	const exhausted = hit.not().and( s.lessThan( 1 ) ).toVar();
	const rejected = bool( false ).toVar();
	If( hit.and( validated.not() ), () => {

		hit.assign( acceptHit( hitUV, hitS, hitDepth ) );
		rejected.assign( hit.not() );

	} );

	return { hit, uv: hitUV, depth: hitDepth, parameter: hitS, iterations, exhausted, rejected };

}

/**
 * Traces screen-space rays through a hierarchical depth buffer it builds from the depth, see `traceHiZ()`.
 * The cost of a ray barely depends on its length. Requires a WebGPU backend.
 *
 * @three_import import { HiZTracer } from 'three/addons/tsl/utils/traceHiZ.js';
 */
export class HiZTracer {

	/**
	 * Constructs a new Hi-Z tracer.
	 *
	 * @param {Object} [options] - The options.
	 * @param {boolean} [options.anyHit=false] - Whether a ray stops at any accepted surface instead of the closest one.
	 */
	constructor( { anyHit = false } = {} ) {

		/**
		 * Whether a ray stops at any accepted surface instead of the closest one, read when the tracing is built.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.anyHit = anyHit;

		/**
		 * The maximum number of steps of a ray.
		 *
		 * @type {UniformNode<int>}
		 * @default 64
		 */
		this.maxSteps = uniform( 64, 'int' );

		this._hiZNode = null;

	}

	_getHiZNode( depthNode ) {

		if ( this._hiZNode === null ) this._hiZNode = hiZ( depthNode, { nearest: true } );

		return this._hiZNode;

	}

	/**
	 * Updates the hierarchical depth, called once per frame before tracing.
	 *
	 * @param {NodeFrame} frame - The current node frame.
	 * @param {TextureNode} depthNode - The depth of the scene.
	 */
	update( frame, depthNode ) {

		this._getHiZNode( depthNode ).updateBefore( frame );

	}

	/**
	 * Traces a ray inside the calling TSL function.
	 *
	 * @param {TextureNode} depthNode - The depth of the scene.
	 * @param {Node<vec3>} origin - Start UV and projected depth.
	 * @param {Node<vec3>} end - End UV and projected depth.
	 * @param {Object} options - The camera's near and far values and the predicates, see `traceHiZ()`.
	 * @return {Object} Hit flag, UV, raw depth and parameter.
	 */
	trace( depthNode, origin, end, options ) {

		const hiZNode = this._getHiZNode( depthNode );

		return traceHiZ( hiZNode, origin, end, {
			...options,
			start: end.xy.sub( origin.xy ).mul( hiZNode.size ).length().max( 1 ).reciprocal(), // skip the pixel of the origin
			maxSteps: this.maxSteps,
			depthTolerance: 2e-7,
			anyHit: this.anyHit
		} );

	}

	dispose() {

		if ( this._hiZNode !== null ) this._hiZNode.dispose();

	}

}

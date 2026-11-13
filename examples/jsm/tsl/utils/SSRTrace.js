import { Break, Continue, If, Loop, bool, float, int, max, min, uint, uvec2, vec2, textureSize } from 'three/tsl';

/**
 * Builds hierarchical screen-space traversal inside the calling TSL function.
 * UV and projected device depth are affine in the same ray parameter, [0,1].
 * The endpoints must already be clipped to the viewport and camera near plane.
 *
 * Cells are skipped only while the ray is in front of their nearest surface.
 * Potential intersections descend to mip zero. Both depth directions are solved,
 * including a ray approaching a surface from behind while moving toward the camera.
 * A caller-supplied predicate performs full-resolution normal/thickness validation;
 * The candidate budget controls how many rejected surfaces may be searched.
 * A budget of one validates once after traversal, outside its loop.
 *
 * Derived from the adaptive traversal in three.js PR #35228. Kept separate from
 * ray generation and shading so deterministic GPU tests exercise the production code.
 *
 * @param {HiZNode} hierarchy - Nearest-depth pyramid of the scene being traced.
 * @param {Node<vec3>} origin - Start UV and projected depth.
 * @param {Node<vec3>} end - End UV and projected depth.
 * @param {Object} options - Traversal configuration.
 * @param {boolean} [options.reversedDepth=false] - Projected depth convention.
 * @param {Node<float>|number} [options.start=0] - Initial ray parameter (self-hit exclusion).
 * @param {Node<int>|number} [options.maxSteps=32] - Work budget, independent of legacy SSR quality.
 * @param {Function} [options.projectDepth] - Converts raw hierarchy depth to projected depth (log-depth inputs).
 * @param {Node<float>|number} [options.depthTolerance=0] - Conservative projected-depth rounding tolerance.
 * @param {number} [options.maxCandidates=1] - Maximum candidate surfaces to validate (compile-time integer).
 * @param {Node<int>|number} [options.minLevel=0] - Finest level later descents may reach (experimental; FidelityFX's most_detailed_mip). Coarse leaves do less work but, with a yes/no validator, their false candidates use up the budget: pair with maxCandidates > 1.
 * @param {boolean} [options.leafSample=false] - Approximate hits: traversal starts at minLevel, and a leaf above level zero tests only the pixel under the ray, as one linear march step does. Surfaces elsewhere in the cell are stepped over.
 * @param {Function} [options.isHidden] - TSL predicate (entry UV, exit UV, raw farthest depth): true when no surface in the cell can be accepted because the ray is too far behind all of them. Such cells are skipped like empty ones, without spending candidates.
 * @param {Function} [options.acceptHit] - TSL predicate accepting (UV, parameter, raw depth).
 * @return {Object} Hit flag, UV, raw depth, parameter, iterations and exhausted flag.
 */
export function traceHiZ( hierarchy, origin, end, { reversedDepth = false, start = 0, maxSteps = 32, projectDepth = value => value, depthTolerance = 0, maxCandidates = 1, minLevel = 0, leafSample = false, isHidden = null, acceptHit = () => bool( true ) } = {} ) {

	if ( ! Number.isInteger( maxCandidates ) || maxCandidates < 1 || maxCandidates > 4096 ) throw new Error( 'SSRTrace: maxCandidates must be an integer from 1 to 4096.' );

	const delta = end.sub( origin ).toConst();
	const s = float( start ).toVar();
	const coarse = minLevel !== 0;
	const minMip = int( minLevel ).min( int( hierarchy.getLastLevelNode() ) ).toConst();
	// Traversal starts at mip zero: the cells next to the origin hold its own
	// surface, which only full resolution separates from the ray. Leaf sampling
	// tests the ray's own pixel at every leaf, so it can start at the leaf level.
	const mip = ( leafSample ? minMip : int( 0 ) ).toVar();
	// A coarse leaf holds its cell's nearest depth. Candidates are validated
	// against the full-resolution depth at their own UV instead.
	const leafDepth = ( uv, raw ) => coarse && ! leafSample ? hierarchy.getTextureNode( 0, uv ) : raw;
	const lastMip = int( hierarchy.getLastLevelNode() ).toConst();
	const hit = bool( false ).toVar();
	const hitUV = vec2( 0 ).toVar();
	const hitDepth = float( 0 ).toVar();
	const hitS = float( 0 ).toVar();
	const iterations = int( 0 ).toVar();
	const candidates = int( 0 ).toVar();
	const size = vec2( textureSize( hierarchy.depthNode ) ).toConst();
	// Traversal runs in level-zero pixels: cells are integer shifts of them.
	const originPixel = origin.xy.mul( size ).toConst();
	const deltaPixel = delta.xy.mul( size ).toConst();
	const pixels = deltaPixel.abs();
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
	const farther = reversedDepth ? ( a, b ) => a.lessThan( b ) : ( a, b ) => a.greaterThan( b );
	const nearer = reversedDepth ? ( a, b ) => a.greaterThan( b ) : ( a, b ) => a.lessThan( b );

	Loop( { start: int( 0 ), end: int( maxSteps ).clamp( 1, 4096 ), type: 'int', condition: '<' }, ( { i } ) => {

		If( s.greaterThanEqual( 1 ), () => {

			Break();

		} );
		iterations.assign( i.add( 1 ) );
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

		const raw = cellDepth.toVar();
		const depth = projectDepth( raw ).toVar();
		const plane = depth.sub( origin.z ).mul( inverseZ ).toVar();
		const conservativeDepth = depth.add( reversedDepth ? depthTolerance : float( depthTolerance ).negate() );
		const inFront = nearer( z, conservativeDepth ).and(
			farther( end.z, origin.z ).not().or( plane.greaterThan( s.add( 1e-7 ) ) )
		).toVar();
		const background = reversedDepth ? raw.lessThanEqual( 0 ) : raw.greaterThanEqual( 1 );

		// Behind every surface in the cell from entry to exit: the ray is hidden
		// by an occluder here and cannot be accepted, so skip it as if empty.
		const hidden = bool( false ).toVar();
		if ( isHidden !== null ) {

			const farRaw = cellFarthest.toVar();
			const far = projectDepth( farRaw ).add( reversedDepth ? float( depthTolerance ).negate() : depthTolerance );
			If( inFront.not().and( background.not() ).and( farther( z, far ) ).and( farther( origin.z.add( delta.z.mul( exit ) ), far ) ), () => {

				hidden.assign( isHidden( pixel.div( size ), origin.xy.add( delta.xy.mul( exit ) ), farRaw ) );

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

				s.assign( max( exit.add( epsilon ), s.add( epsilon ) ) );
				// Ascend only when leaving the parent cell. Inside the same parent
				// the coarser test would repeat and cost a second read to descend.
				const leavesParent = cell.bitAnd( uvec2( 1 ) ).equal( exitParity );
				If( exits.x.lessThanEqual( exits.y ).select( leavesParent.x, leavesParent.y ), () => {

					mip.assign( mip.add( 1 ).min( lastMip ) );

				} );
				Continue();

			} );

		} );

		If( mip.greaterThan( minMip ), () => {

			mip.subAssign( 1 );
			Continue();

		} );

		if ( leafSample ) {

			// Approximate leaf: like one linear march step, test only the pixel
			// under the ray. Surfaces elsewhere in the cell are stepped over.
			If( mip.greaterThan( 0 ), () => {

				const fine = hierarchy.getPixelCellNode( 0, pixel ).depth.toVar();
				If( nearer( z, projectDepth( fine ) ).or( reversedDepth ? fine.lessThanEqual( 0 ) : fine.greaterThanEqual( 1 ) ), () => {

					s.assign( max( exit.add( epsilon ), s.add( epsilon ) ) );
					const leavesParent = cell.bitAnd( uvec2( 1 ) ).equal( exitParity );
					If( exits.x.lessThanEqual( exits.y ).select( leavesParent.x, leavesParent.y ), () => {

						mip.assign( mip.add( 1 ).min( lastMip ) );

					} );
					Continue();

				} );
				raw.assign( fine );

			} );

		}

		candidates.addAssign( 1 );
		if ( maxCandidates === 1 ) {

			// AMD-style termination: return the first geometric candidate, then
			// validate it once outside the traversal loop.
			hit.assign( true );
			hitUV.assign( pixel.div( size ) );
			hitDepth.assign( raw );
			hitS.assign( s );
			Break();

		} else {

			// At a leaf, test entry first. If it is too far behind the surface,
			// a camera-facing ray can still reach its depth plane inside this cell.
			const candidate = s.toVar();
			const candidateUV = pixel.div( size ).toVar();
			const candidateDepth = float( leafDepth( candidateUV, raw ) ).toVar();
			const accepted = bool( acceptHit( candidateUV, candidate, candidateDepth ) ).toVar();
			If( accepted.not().and( nearer( end.z, origin.z ) ).and( plane.greaterThan( s ) ).and( plane.lessThan( exit ) ), () => {

				candidate.assign( plane );
				candidateUV.assign( origin.xy.add( delta.xy.mul( plane ) ) );
				candidateDepth.assign( leafDepth( candidateUV, raw ) );
				accepted.assign( acceptHit( candidateUV, candidate, candidateDepth ) );

			} );
			If( accepted, () => {

				hit.assign( true );
				hitUV.assign( candidateUV );
				hitDepth.assign( candidateDepth );
				hitS.assign( candidate );
				Break();

			} );
			If( candidates.greaterThanEqual( maxCandidates ), () => {

				s.assign( 1 );
				Break();

			} );
			s.assign( max( exit.add( epsilon ), s.add( epsilon ) ) );
			// Stay at the leaf after rejection; ascend once a cell is proven empty.

		}

	} );

	const candidate = candidates.greaterThan( 0 );
	const exhausted = hit.not().and( s.lessThan( 1 ) ).toVar();
	if ( maxCandidates === 1 ) {

		If( hit, () => {

			if ( coarse && ! leafSample ) hitDepth.assign( leafDepth( hitUV ) );
			hit.assign( acceptHit( hitUV, hitS, hitDepth ) );

		} );

	}

	return { candidates, candidate, hit, uv: hitUV, depth: hitDepth, parameter: hitS, iterations, exhausted };

}

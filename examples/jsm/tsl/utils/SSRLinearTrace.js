import { Break, If, Loop, bool, float, int, vec2 } from 'three/tsl';

/**
 * Builds SSRNode's original fixed-schedule depth march and binary refinement.
 * Kept independent of ray generation/shading so tests can feed the identical
 * explicit rays to the production linear and hierarchical algorithms.
 *
 * Callbacks retain SSRNode's existing projection, thickness, normal and range
 * rules. acceptHit runs inside the march and can Continue/Break for rejection.
 * The returned depth is projected depth, matching SSRNode's shading contract.
 *
 * @param {Object} options - Production sampling and hit-validation callbacks.
 * @return {Object} Hit flag, UV, projected depth, ray parameter and iterations.
 */
export function traceLinearSSR( { totalStep, sampleFraction, uvAt, sampleDepth, projectDepth, sceneZ, rayZAt, isBackground, acceptHit, frontFacing, binaryRefine = false } ) {

	const hit = bool( false ).toVar();
	const hitUV = vec2( 0 ).toVar();
	const hitDepth = float( 0 ).toVar();
	const hitS = float( 0 ).toVar();
	const previousS = float( 0 ).toVar();
	const previousInFront = bool( true ).toVar();
	const canRefine = bool( false ).toVar();
	const lo = float( 0 ).toVar();
	const hi = float( 0 ).toVar();
	const iterations = int( 0 ).toVar();

	Loop( { start: int( 1 ), end: totalStep }, ( { i } ) => {

		const s = sampleFraction( float( i ) ).toVar();
		If( s.greaterThan( 1 ), () => {

			Break();

		} );
		const uv = uvAt( s ).toVar();
		If( uv.lessThan( vec2( 0 ) ).any().or( uv.greaterThan( vec2( 1 ) ).any() ), () => {

			Break();

		} );
		iterations.addAssign( 1 );
		const raw = sampleDepth( uv ).toVar();
		const depth = projectDepth( raw ).toVar();
		const z = sceneZ( raw ).toVar();
		const rayZ = rayZAt( s ).toVar();
		const lastS = previousS.toVar();
		const lastInFront = previousInFront.toVar();
		previousS.assign( s );
		previousInFront.assign( rayZ.greaterThan( z ) );

		If( isBackground( depth ).not().and( rayZ.lessThanEqual( z ) ), () => {

			If( acceptHit( uv, s, lastS, depth ), () => {

				hit.assign( true );
				hitUV.assign( uv );
				hitDepth.assign( depth );
				hitS.assign( s );
				if ( binaryRefine ) {

					canRefine.assign( lastInFront );
					lo.assign( lastS );
					hi.assign( s );

				}

				Break();

			} );

		} );

	} );

	// Refine after marching, preserving the original avoidance of nested loops
	// (some drivers miscompiled a refinement loop inside the marching loop).
	if ( binaryRefine ) {

		If( hit.and( canRefine ), () => {

			Loop( { start: int( 0 ), end: int( 8 ), type: 'int', condition: '<' }, () => {

				const mid = lo.add( hi ).mul( 0.5 ).toVar();
				const uv = uvAt( mid ).toConst();
				If( rayZAt( mid ).lessThanEqual( sceneZ( sampleDepth( uv ) ) ).and( frontFacing( uv ) ), () => {

					hi.assign( mid );

				} ).Else( () => {

					lo.assign( mid );

				} );

			} );
			hitS.assign( hi );
			hitUV.assign( uvAt( hi ) );
			hitDepth.assign( projectDepth( sampleDepth( hitUV ) ) );

		} );

	}

	return { hit, uv: hitUV, depth: hitDepth, parameter: hitS, iterations };

}

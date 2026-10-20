import { Vector2 } from 'three/webgpu';
import { Fn, If, Loop, float, vec2, ivec2, uniform, reference, renderGroup, texture, textureLoad, screenCoordinate, interleavedGradientNoise, vogelDiskSample, perspectiveDepthToViewZ, dFdx, dFdy, abs, max } from 'three/tsl';

/**
 * Example-only PCSS filter for a spot light with conventional perspective depth.
 * lightRadius is the emitter radius in world units; maxRadius is in shadow texels.
 * Sample counts are adjustable integer uniforms (blockers: 1–64, filter: 1–128).
 * Blocker depths are linearized before averaging. Multiple blocker layers are
 * approximated by one plane, so overlapping penumbrae can leak light.
 *
 * @param {SpotLight} light - The shadow-casting spot light.
 * @return {Object} The filter and its adjustable uniforms.
 */
export function spotLightPCSS( light ) {

	const lightRadius = uniform( 0.35 );
	const maxRadius = uniform( 64 );
	const frame = uniform( 0 );
	const blockerSamples = uniform( 16, 'int' );
	const filterSamples = uniform( 32, 'int' );
	const shadow = light.shadow;
	const near = reference( 'near', 'float', shadow.camera ).setGroup( renderGroup );
	const far = reference( 'far', 'float', shadow.camera ).setGroup( renderGroup );
	const mapSize = reference( 'mapSize', 'vec2', shadow ).setGroup( renderGroup );
	const projectionScale = uniform( new Vector2() ).onRenderUpdate( () => {

		const elements = shadow.camera.projectionMatrix.elements;
		return projectionScale.value.set( elements[ 0 ], elements[ 5 ] ).multiplyScalar( 0.5 );

	} );

	const filterNode = Fn( ( { depthTexture, shadowCoord }, builder ) => {

		if ( builder.renderer.logarithmicDepthBuffer ) throw new Error( 'Spot light PCSS requires perspective shadow depth.' );

		const texel = vec2( 1 ).div( mapSize );
		const lower = texel.mul( 0.5 );
		const upper = lower.oneMinus();
		const receiver = perspectiveDepthToViewZ( shadowCoord.z, near, far ).negate();
		const emitterUV = lightRadius.mul( projectionScale );
		const phi = interleavedGradientNoise( screenCoordinate.xy ).add( frame.mul( 0.61803398875 ) ).fract().mul( Math.PI * 2 );

		// Solve dz/du and dz/dv before entering non-uniform blocker branches.
		const dx = dFdx( shadowCoord ).toVar();
		const dy = dFdy( shadowCoord ).toVar();
		const determinant = dx.x.mul( dy.y ).sub( dx.y.mul( dy.x ) );
		const safeDeterminant = abs( determinant ).greaterThan( 1e-10 ).select( determinant, 1 );
		const gradient = vec2( dy.y.mul( dx.z ).sub( dx.y.mul( dy.z ) ), dx.x.mul( dy.z ).sub( dy.x.mul( dx.z ) ) ).div( safeDeterminant ).toVar();
		const compareDepth = ( offset ) => shadowCoord.z.add( gradient.dot( offset ) );

		// Project the emitter through the near plane onto the receiver plane.
		const searchRadius = emitterUV.mul( receiver.sub( near ) ).div( receiver.mul( near ) ).min( maxRadius.mul( texel ) );
		const blockerSampleCount = blockerSamples.clamp( 1, 64 ).toVar();
		const filterSampleCount = filterSamples.clamp( 1, 128 ).toVar();
		const blockerSum = float( 0 ).toVar();
		const blockerCount = float( 0 ).toVar();

		// Include the central texel: sparse disk taps can miss thin contact blockers.
		const centerUV = shadowCoord.xy.clamp( lower, upper );
		const addBlocker = ( uv ) => {

			const pixel = ivec2( uv.mul( mapSize ) );
			// textureLoad reads a texel center, not the continuous disk-sample UV.
			const sampleUV = vec2( pixel ).add( 0.5 ).mul( texel );
			const depth = textureLoad( depthTexture, pixel ).r;
			const isBlocker = builder.renderer.reversedDepthBuffer ? depth.greaterThan( compareDepth( sampleUV.sub( shadowCoord.xy ) ) ) : depth.lessThan( compareDepth( sampleUV.sub( shadowCoord.xy ) ) );
			If( isBlocker, () => {

				blockerSum.addAssign( perspectiveDepthToViewZ( depth, near, far ).negate() );
				blockerCount.addAssign( 1 );

			} );

		};

		addBlocker( centerUV );

		// The center is included in the requested blocker-sample budget.
		Loop( { start: 1, end: blockerSampleCount, type: 'int', condition: '<' }, ( { i } ) => {

			const uv = shadowCoord.xy.add( vogelDiskSample( i.sub( 1 ), blockerSampleCount.sub( 1 ), phi ).mul( searchRadius ) ).clamp( lower, upper );
			addBlocker( uv );

		} );

		// If the wide search misses blockers, retain the local shadow instead of
		// declaring the receiver fully lit. This also preserves hardware PCF edges.
		const visibility = texture( depthTexture, centerUV ).compare( shadowCoord.z ).toVar();
		If( blockerCount.greaterThan( 0 ), () => {

			const blocker = blockerSum.div( blockerCount );
			// World penumbra radius = lightRadius * (receiver - blocker) / blocker.
			// Project onto shadow UV at the receiver, not at the near plane.
			const radius = emitterUV.mul( max( receiver.sub( blocker ), 0 ) ).div( blocker.mul( receiver ) ).clamp( texel, maxRadius.mul( texel ) );
			visibility.assign( 0 );
			Loop( { start: 0, end: filterSampleCount, type: 'int', condition: '<' }, ( { i } ) => {

				const uv = shadowCoord.xy.add( vogelDiskSample( i, filterSampleCount, phi ).mul( radius ) ).clamp( lower, upper );
				visibility.addAssign( texture( depthTexture, uv ).compare( compareDepth( uv.sub( shadowCoord.xy ) ) ) );

			} );
			visibility.divAssign( float( filterSampleCount ) );

		} );
		return visibility;

	} );

	return { filterNode, lightRadius, maxRadius, frame, blockerSamples, filterSamples };

}


import { cross, dot, max } from 'three/tsl';

// These predicates intentionally describe different acceptance volumes.
// Keep their geometry separate from traversal and lighting/denoising.

export function getSSRPixelWidth( inverseProjection, viewZ, inverseWidth, perspective ) {

	const width = inverseProjection.element( 0 ).x.mul( 2 ).mul( inverseWidth );
	return perspective ? width.mul( viewZ.negate() ) : width;

}

export function validateSSRRayDistance( point, origin, direction, thickness ) {

	// Equivalent to point-to-line distance <= thickness, without square roots
	// or division. The direction need not be normalized.
	const perpendicular = cross( point.sub( origin ), direction ).toVar();
	return dot( perpendicular, perpendicular ).lessThanEqual( thickness.mul( thickness ).mul( dot( direction, direction ) ) );

}

export function validateSSRViewDistance( surface, rayPoint, thickness ) {

	const difference = surface.sub( rayPoint ).toVar();
	return dot( difference, difference ).lessThanEqual( thickness.mul( thickness ) );

}

export function validateSSRDepthSlab( rayZ, surfaceZ, relativeThickness ) {

	// View-space Z is negative. Include the surface and the back of the slab.
	const tolerance = surfaceZ.abs().max( 1 ).mul( 1e-6 );
	return rayZ.lessThanEqual( surfaceZ.add( tolerance ) ).and( rayZ.greaterThanEqual( surfaceZ.mul( max( relativeThickness, 0 ).add( 1 ) ).sub( tolerance ) ) );

}

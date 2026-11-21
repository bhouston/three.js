import { cross, dot } from 'three/tsl';

// Keep the original SSR acceptance geometry separate from traversal and shading.

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

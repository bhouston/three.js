import { Fn, float, frameId, screenCoordinate, uint } from 'three/tsl';

const hashUint = Fn( ( [ input ] ) => {

	const value = uint( input ).toVar();
	value.assign( value.bitXor( value.shiftRight( 16 ) ).mul( uint( 0x21f0aaad ) ) );
	value.assign( value.bitXor( value.shiftRight( 15 ) ).mul( uint( 0x735a2d97 ) ) );
	return value.bitXor( value.shiftRight( 15 ) );

} );

const pixelHash = ( x, y, seed ) => hashUint( x.mul( uint( 0x9e3779b1 ) ).bitXor( y.mul( uint( 0x85ebca77 ) ) ).bitXor( seed ) );

/**
 * Three-level hierarchy + Weyl (3LHW). Each aligned 8×8 tile has a scrambled
 * hierarchical rank pattern, a hashed phase, and a golden-ratio temporal shift.
 * This inexpensive scalar stratified sequence is not an optimized FAST mask.
 *
 * @param {Node<vec2>} [coordinate=screenCoordinate] - Integer pixel coordinates.
 * @param {Node<uint>} [index=frameId] - Temporal sample index.
 * @param {Node<uint>|number} [seed=1] - Fixed random seed.
 * @return {Node<float>} Scalar sample in [0,1).
 */
export const threeLevelHierarchyWeyl = ( coordinate = screenCoordinate, index = frameId, seed = 1 ) => Fn( () => {

	const x = uint( coordinate.x ), y = uint( coordinate.y ), s = uint( seed );
	const rank = uint( 0 ).toVar();
	for ( let level = 0; level < 3; level ++ ) {

		const scramble = pixelHash( x.shiftRight( level + 1 ), y.shiftRight( level + 1 ), s.bitXor( uint( ( Math.imul( level + 1, 0x632be5ab ) ) >>> 0 ) ) );
		const a = x.shiftRight( level ).bitAnd( 1 );
		const b = y.shiftRight( level ).bitAnd( 1 );
		const low = scramble.bitAnd( 4 ).notEqual( 0 ).select( a, b );
		const child = a.bitXor( b ).shiftLeft( 1 ).bitOr( low );
		rank.assign( rank.bitOr( child.bitXor( scramble.bitAnd( 3 ) ).shiftLeft( 30 - 2 * level ) ) );

	}

	const phase = pixelHash( x.shiftRight( 3 ), y.shiftRight( 3 ), s.bitXor( uint( 0x02e5be93 ) ) );
	return rank.add( phase ).add( uint( index ).mul( uint( 2654435769 ) ) ).shiftRight( 8 ).toFloat().div( 16777216 );

} )();

/**
 * Animated interleaved gradient noise (Jimenez), with the 5.588238-pixel
 * diagonal temporal scroll described by Jorge Jimenez / Demofox. No texture
 * or startup generation is required. Floating-point precision limits very
 * large coordinates and frame indices; reset the index when history resets.
 *
 * @param {Node<vec2>} [coordinate=screenCoordinate] - Pixel coordinates.
 * @param {Node<uint>} [index=frameId] - Temporal sample index.
 * @param {Node<uint>|number} [seed=1] - Fixed random seed.
 * @return {Node<float>} Scalar sample in [0,1).
 */
export const animatedIGN = ( coordinate = screenCoordinate, index = frameId, seed = 1 ) => Fn( () => {

	const shift = float( index ).mul( 5.588238 );
	const offset = hashUint( uint( seed ) ).shiftRight( 8 ).toFloat().div( 16777216 ).mul( 256 );
	const px = float( coordinate.x ).add( shift ).add( offset );
	const py = float( coordinate.y ).add( shift ).add( offset );
	return px.mul( 0.06711056 ).add( py.mul( 0.00583715 ) ).fract().mul( 52.9829189 ).fract();

} )();

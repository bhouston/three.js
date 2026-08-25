import { describe, expect, it } from 'vitest';
import { IDENTITY_UV_TRANSFORM, composeUvTransformMatrix, resolveUvTransformMatrix } from '../../../../examples/jsm/ntc/NTCUvTransform.js';

function applyMatrix( matrix, u, v ) {

	const [ a, b, c, d, e, f ] = matrix;

	return [ a * u + b * v + e, c * u + d * v + f ];

}

describe( 'Addons > NTC > NTCUvTransform', () => {

	describe( 'composeUvTransformMatrix', () => {

		it( 'produces the identity matrix for rotation 0, scale [1, 1]', () => {

			expect( composeUvTransformMatrix( 0, [ 1, 1 ] ) ).toEqual( IDENTITY_UV_TRANSFORM );

		} );

		it( 'maps a coordinate through a pure 90deg rotation as expected', () => {

			const matrix = composeUvTransformMatrix( Math.PI / 2, [ 1, 1 ] );
			const [ u, v ] = applyMatrix( matrix, 1, 0 );

			expect( u ).toBeCloseTo( 0, 10 );
			expect( v ).toBeCloseTo( 1, 10 );

		} );

		it( 'maps a coordinate through a pure anisotropic scale as expected', () => {

			const matrix = composeUvTransformMatrix( 0, [ 2, 3 ] );
			const [ u, v ] = applyMatrix( matrix, 1, 1 );

			expect( u ).toBeCloseTo( 2, 10 );
			expect( v ).toBeCloseTo( 3, 10 );

		} );

		it( 'composes scale-then-rotate (M = R . S), not rotate-then-scale', () => {

			// Scaling x by 2 then rotating 90deg should send (1,0) -> (0,2),
			// not (0,1) scaled some other way - distinguishes composition order.
			const matrix = composeUvTransformMatrix( Math.PI / 2, [ 2, 1 ] );
			const [ u, v ] = applyMatrix( matrix, 1, 0 );

			expect( u ).toBeCloseTo( 0, 10 );
			expect( v ).toBeCloseTo( 2, 10 );

		} );

		it( 'defaults translation to [0, 0] when omitted', () => {

			const matrix = composeUvTransformMatrix( 0.3, [ 1.5, 0.5 ] );

			expect( matrix[ 4 ] ).toBe( 0 );
			expect( matrix[ 5 ] ).toBe( 0 );

		} );

		it( 'includes an explicit translation when given', () => {

			const matrix = composeUvTransformMatrix( 0, [ 1, 1 ], [ 0.25, - 0.1 ] );

			expect( matrix[ 4 ] ).toBe( 0.25 );
			expect( matrix[ 5 ] ).toBe( - 0.1 );

		} );

	} );

	describe( 'resolveUvTransformMatrix', () => {

		it( 'passes null/undefined through as null', () => {

			expect( resolveUvTransformMatrix( null ) ).toBeNull();
			expect( resolveUvTransformMatrix( undefined ) ).toBeNull();

		} );

		it( 'passes an already-baked flat matrix through unchanged', () => {

			const matrix = [ 1, 2, 3, 4, 5, 6 ];

			expect( resolveUvTransformMatrix( matrix ) ).toBe( matrix );

		} );

		it( 'composes a decomposed { rotation, scale } object on demand - the shape a live mid-training onProgress snapshot has (see NTCTrainer.js)', () => {

			const resolved = resolveUvTransformMatrix( { rotation: 0, scale: [ 1, 1 ] } );

			expect( resolved ).toEqual( composeUvTransformMatrix( 0, [ 1, 1 ] ) );
			expect( resolved ).toEqual( IDENTITY_UV_TRANSFORM );

		} );

	} );

} );

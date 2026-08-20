import { describe, expect, it } from 'vitest';
import { cross, dot, normalize } from '../../../../examples/jsm/neural/NeuralVectorMath.js';

describe( 'Addons > Neural > NeuralVectorMath', () => {

	describe( 'dot', () => {

		it( 'computes the dot product of two vectors', () => {

			expect( dot( [ 1, 2, 3 ], [ 4, 5, 6 ] ) ).toBe( 1 * 4 + 2 * 5 + 3 * 6 );

		} );

		it( 'is zero for perpendicular unit axes', () => {

			expect( dot( [ 1, 0, 0 ], [ 0, 1, 0 ] ) ).toBe( 0 );

		} );

		it( 'equals the squared length for a vector dotted with itself', () => {

			const v = [ 3, 4, 0 ];
			expect( dot( v, v ) ).toBe( 25 );

		} );

		it( 'is symmetric', () => {

			const a = [ 1, - 2, 3 ];
			const b = [ - 4, 5, - 6 ];
			expect( dot( a, b ) ).toBe( dot( b, a ) );

		} );

	} );

	describe( 'cross', () => {

		it( 'computes x cross y = z for the standard basis', () => {

			expect( cross( [ 1, 0, 0 ], [ 0, 1, 0 ] ) ).toEqual( [ 0, 0, 1 ] );

		} );

		it( 'computes y cross z = x', () => {

			expect( cross( [ 0, 1, 0 ], [ 0, 0, 1 ] ) ).toEqual( [ 1, 0, 0 ] );

		} );

		it( 'anticommutes: a x b = -(b x a)', () => {

			const a = [ 1, 2, 3 ];
			const b = [ 4, - 5, 6 ];
			const ab = cross( a, b );
			const ba = cross( b, a );

			expect( ab ).toEqual( [ - ba[ 0 ], - ba[ 1 ], - ba[ 2 ] ] );

		} );

		it( 'is zero for a vector crossed with itself', () => {

			expect( cross( [ 2, - 3, 5 ], [ 2, - 3, 5 ] ) ).toEqual( [ 0, 0, 0 ] );

		} );

		it( 'is perpendicular to both inputs', () => {

			const a = [ 1, 2, 3 ];
			const b = [ - 1, 0, 4 ];
			const n = cross( a, b );

			expect( dot( n, a ) ).toBeCloseTo( 0, 10 );
			expect( dot( n, b ) ).toBeCloseTo( 0, 10 );

		} );

	} );

	describe( 'normalize', () => {

		it( 'scales a vector to unit length', () => {

			const result = normalize( [ 3, 4, 0 ] );
			expect( result[ 0 ] ).toBeCloseTo( 0.6, 10 );
			expect( result[ 1 ] ).toBeCloseTo( 0.8, 10 );
			expect( result[ 2 ] ).toBeCloseTo( 0, 10 );
			expect( Math.hypot( ...result ) ).toBeCloseTo( 1, 10 );

		} );

		it( 'leaves an already-unit vector unchanged', () => {

			const result = normalize( [ 0, 1, 0 ] );
			expect( result ).toEqual( [ 0, 1, 0 ] );

		} );

		it( 'does not divide by zero for the zero vector (falls back to length 1)', () => {

			const result = normalize( [ 0, 0, 0 ] );

			expect( result ).toEqual( [ 0, 0, 0 ] );
			expect( result.every( Number.isFinite ) ).toBe( true );

		} );

		it( 'normalizes a tiny vector without producing NaN/Infinity', () => {

			const result = normalize( [ 1e-20, 0, 0 ] );

			expect( result.every( Number.isFinite ) ).toBe( true );

		} );

		it( 'normalizes negative-component vectors correctly', () => {

			const result = normalize( [ - 3, - 4, 0 ] );
			expect( Math.hypot( ...result ) ).toBeCloseTo( 1, 10 );
			expect( result[ 0 ] ).toBeLessThan( 0 );
			expect( result[ 1 ] ).toBeLessThan( 0 );

		} );

	} );

} );

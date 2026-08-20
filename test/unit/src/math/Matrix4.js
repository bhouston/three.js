import { describe, test, expect } from 'vitest';
import { Matrix3 } from '@src/math/Matrix3.js';
import { Matrix4 } from '@src/math/Matrix4.js';
import { Vector3 } from '@src/math/Vector3.js';
import { Euler } from '@src/math/Euler.js';
import { Quaternion } from '@src/math/Quaternion.js';
import * as MathUtils from '@src/math/MathUtils.js';
import { eps } from '@test-utils/math-constants.js';

function matrixEquals4( a, b, tolerance ) {

	tolerance = tolerance || 0.0001;
	if ( a.elements.length != b.elements.length ) {

		return false;

	}

	for ( let i = 0, il = a.elements.length; i < il; i ++ ) {

		const delta = Math.abs( a.elements[ i ] - b.elements[ i ] );
		if ( delta > tolerance ) {

			return false;

		}

	}

	return true;

}

// from Euler.js
function eulerEquals( a, b, tolerance ) {

	tolerance = tolerance || 0.0001;
	const diff = Math.abs( a.x - b.x ) + Math.abs( a.y - b.y ) + Math.abs( a.z - b.z );
	return ( diff < tolerance );

}

describe( 'Maths', () => {

	describe( 'Matrix4', () => {

		test( 'Instancing', () => {

			const a = new Matrix4();
			expect( a.determinant() == 1 ).toBeTruthy();

			const b = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			expect( b.elements[ 0 ] == 0 ).toBeTruthy();
			expect( b.elements[ 1 ] == 4 ).toBeTruthy();
			expect( b.elements[ 2 ] == 8 ).toBeTruthy();
			expect( b.elements[ 3 ] == 12 ).toBeTruthy();
			expect( b.elements[ 4 ] == 1 ).toBeTruthy();
			expect( b.elements[ 5 ] == 5 ).toBeTruthy();
			expect( b.elements[ 6 ] == 9 ).toBeTruthy();
			expect( b.elements[ 7 ] == 13 ).toBeTruthy();
			expect( b.elements[ 8 ] == 2 ).toBeTruthy();
			expect( b.elements[ 9 ] == 6 ).toBeTruthy();
			expect( b.elements[ 10 ] == 10 ).toBeTruthy();
			expect( b.elements[ 11 ] == 14 ).toBeTruthy();
			expect( b.elements[ 12 ] == 3 ).toBeTruthy();
			expect( b.elements[ 13 ] == 7 ).toBeTruthy();
			expect( b.elements[ 14 ] == 11 ).toBeTruthy();
			expect( b.elements[ 15 ] == 15 ).toBeTruthy();

			expect( ! matrixEquals4( a, b ) ).toBeTruthy();

			const c = new Matrix4( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			expect( c.elements[ 0 ] == 0 ).toBeTruthy();
			expect( c.elements[ 1 ] == 4 ).toBeTruthy();
			expect( c.elements[ 2 ] == 8 ).toBeTruthy();
			expect( c.elements[ 3 ] == 12 ).toBeTruthy();
			expect( c.elements[ 4 ] == 1 ).toBeTruthy();
			expect( c.elements[ 5 ] == 5 ).toBeTruthy();
			expect( c.elements[ 6 ] == 9 ).toBeTruthy();
			expect( c.elements[ 7 ] == 13 ).toBeTruthy();
			expect( c.elements[ 8 ] == 2 ).toBeTruthy();
			expect( c.elements[ 9 ] == 6 ).toBeTruthy();
			expect( c.elements[ 10 ] == 10 ).toBeTruthy();
			expect( c.elements[ 11 ] == 14 ).toBeTruthy();
			expect( c.elements[ 12 ] == 3 ).toBeTruthy();
			expect( c.elements[ 13 ] == 7 ).toBeTruthy();
			expect( c.elements[ 14 ] == 11 ).toBeTruthy();
			expect( c.elements[ 15 ] == 15 ).toBeTruthy();

			expect( ! matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'isMatrix4', () => {

			const a = new Matrix4();
			expect( a.isMatrix4 === true ).toBeTruthy();

			const b = new Vector3();
			expect( ! b.isMatrix4 ).toBeTruthy();

		} );

		test( 'set', () => {

			const b = new Matrix4();
			expect( b.determinant() == 1 ).toBeTruthy();

			b.set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			expect( b.elements[ 0 ] == 0 ).toBeTruthy();
			expect( b.elements[ 1 ] == 4 ).toBeTruthy();
			expect( b.elements[ 2 ] == 8 ).toBeTruthy();
			expect( b.elements[ 3 ] == 12 ).toBeTruthy();
			expect( b.elements[ 4 ] == 1 ).toBeTruthy();
			expect( b.elements[ 5 ] == 5 ).toBeTruthy();
			expect( b.elements[ 6 ] == 9 ).toBeTruthy();
			expect( b.elements[ 7 ] == 13 ).toBeTruthy();
			expect( b.elements[ 8 ] == 2 ).toBeTruthy();
			expect( b.elements[ 9 ] == 6 ).toBeTruthy();
			expect( b.elements[ 10 ] == 10 ).toBeTruthy();
			expect( b.elements[ 11 ] == 14 ).toBeTruthy();
			expect( b.elements[ 12 ] == 3 ).toBeTruthy();
			expect( b.elements[ 13 ] == 7 ).toBeTruthy();
			expect( b.elements[ 14 ] == 11 ).toBeTruthy();
			expect( b.elements[ 15 ] == 15 ).toBeTruthy();

		} );

		test( 'identity', () => {

			const b = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			expect( b.elements[ 0 ] == 0 ).toBeTruthy();
			expect( b.elements[ 1 ] == 4 ).toBeTruthy();
			expect( b.elements[ 2 ] == 8 ).toBeTruthy();
			expect( b.elements[ 3 ] == 12 ).toBeTruthy();
			expect( b.elements[ 4 ] == 1 ).toBeTruthy();
			expect( b.elements[ 5 ] == 5 ).toBeTruthy();
			expect( b.elements[ 6 ] == 9 ).toBeTruthy();
			expect( b.elements[ 7 ] == 13 ).toBeTruthy();
			expect( b.elements[ 8 ] == 2 ).toBeTruthy();
			expect( b.elements[ 9 ] == 6 ).toBeTruthy();
			expect( b.elements[ 10 ] == 10 ).toBeTruthy();
			expect( b.elements[ 11 ] == 14 ).toBeTruthy();
			expect( b.elements[ 12 ] == 3 ).toBeTruthy();
			expect( b.elements[ 13 ] == 7 ).toBeTruthy();
			expect( b.elements[ 14 ] == 11 ).toBeTruthy();
			expect( b.elements[ 15 ] == 15 ).toBeTruthy();

			const a = new Matrix4();
			expect( ! matrixEquals4( a, b ) ).toBeTruthy();

			b.identity();
			expect( matrixEquals4( a, b ) ).toBeTruthy();

		} );

		test( 'clone', () => {

			const a = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			const b = a.clone();

			expect( matrixEquals4( a, b ) ).toBeTruthy();

			// ensure that it is a true copy
			a.elements[ 0 ] = 2;
			expect( ! matrixEquals4( a, b ) ).toBeTruthy();

		} );

		test( 'copy', () => {

			const a = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			const b = new Matrix4().copy( a );

			expect( matrixEquals4( a, b ) ).toBeTruthy();

			// ensure that it is a true copy
			a.elements[ 0 ] = 2;
			expect( ! matrixEquals4( a, b ) ).toBeTruthy();

		} );

		test( 'setFromMatrix3', () => {

			const a = new Matrix3().set(
				0, 1, 2,
				3, 4, 5,
				6, 7, 8
			);
			const b = new Matrix4();
			const c = new Matrix4().set(
				0, 1, 2, 0,
				3, 4, 5, 0,
				6, 7, 8, 0,
				0, 0, 0, 1
			);
			b.setFromMatrix3( a );
			expect( b.equals( c ) ).toBeTruthy();

		} );

		test( 'copyPosition', () => {

			const a = new Matrix4().set( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
			const b = new Matrix4().set( 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 16 );

			expect( matrixEquals4( a, b ) ).toBeFalsy();

			b.copyPosition( a );
			expect( matrixEquals4( a, b ) ).toBeTruthy();

		} );

		test( 'makeBasis/extractBasis', () => {

			const identityBasis = [ new Vector3( 1, 0, 0 ), new Vector3( 0, 1, 0 ), new Vector3( 0, 0, 1 ) ];
			const a = new Matrix4().makeBasis( identityBasis[ 0 ], identityBasis[ 1 ], identityBasis[ 2 ] );
			const identity = new Matrix4();
			expect( matrixEquals4( a, identity ) ).toBeTruthy();

			const testBases = [[ new Vector3( 0, 1, 0 ), new Vector3( - 1, 0, 0 ), new Vector3( 0, 0, 1 ) ]];
			for ( let i = 0; i < testBases.length; i ++ ) {

				const testBasis = testBases[ i ];
				const b = new Matrix4().makeBasis( testBasis[ 0 ], testBasis[ 1 ], testBasis[ 2 ] );
				const outBasis = [ new Vector3(), new Vector3(), new Vector3() ];
				b.extractBasis( outBasis[ 0 ], outBasis[ 1 ], outBasis[ 2 ] );
				// check what goes in, is what comes out.
				for ( let j = 0; j < outBasis.length; j ++ ) {

					expect( outBasis[ j ].equals( testBasis[ j ] ) ).toBeTruthy();

				}

				// get the basis out the hard war
				for ( let j = 0; j < identityBasis.length; j ++ ) {

					outBasis[ j ].copy( identityBasis[ j ] );
					outBasis[ j ].applyMatrix4( b );

				}

				// did the multiply method of basis extraction work?
				for ( let j = 0; j < outBasis.length; j ++ ) {

					expect( outBasis[ j ].equals( testBasis[ j ] ) ).toBeTruthy();

				}

			}

		} );

		test( 'makeRotationFromEuler/extractRotation', () => {

			const testValues = [
				new Euler( 0, 0, 0, 'XYZ' ),
				new Euler( 1, 0, 0, 'XYZ' ),
				new Euler( 0, 1, 0, 'ZYX' ),
				new Euler( 0, 0, 0.5, 'YZX' ),
				new Euler( 0, 0, - 0.5, 'YZX' )
			];

			for ( let i = 0; i < testValues.length; i ++ ) {

				const v = testValues[ i ];

				const m = new Matrix4().makeRotationFromEuler( v );

				const v2 = new Euler().setFromRotationMatrix( m, v.order );
				const m2 = new Matrix4().makeRotationFromEuler( v2 );

				expect( matrixEquals4( m, m2, eps ) ).toBeTruthy();
				expect( eulerEquals( v, v2, eps ) ).toBeTruthy();

				const m3 = new Matrix4().extractRotation( m2 );
				const v3 = new Euler().setFromRotationMatrix( m3, v.order );

				expect( matrixEquals4( m, m3, eps ) ).toBeTruthy();
				expect( eulerEquals( v, v3, eps ) ).toBeTruthy();

			}

		} );

		test( 'lookAt', () => {

			const a = new Matrix4();
			const expected = new Matrix4().identity();
			const eye = new Vector3( 0, 0, 0 );
			const target = new Vector3( 0, 1, - 1 );
			const up = new Vector3( 0, 1, 0 );

			a.lookAt( eye, target, up );
			const rotation = new Euler().setFromRotationMatrix( a );
			expect( rotation.x * ( 180 / Math.PI ) ).toNumEqual( 45 );

			// eye and target are in the same position
			eye.copy( target );
			a.lookAt( eye, target, up );
			expect( matrixEquals4( a, expected ) ).toBeTruthy();

			// up and z are parallel
			eye.set( 0, 1, 0 );
			target.set( 0, 0, 0 );
			a.lookAt( eye, target, up );
			expected.set(
				1, 0, 0, 0,
				0, 0.0001, 1, 0,
				0, - 1, 0.0001, 0,
				0, 0, 0, 1
			);
			expect( matrixEquals4( a, expected ) ).toBeTruthy();

		} );

		test( 'multiply', () => {

			const lhs = new Matrix4().set( 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 );
			const rhs = new Matrix4().set( 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131 );

			lhs.multiply( rhs );

			expect( lhs.elements[ 0 ] == 1585 ).toBeTruthy();
			expect( lhs.elements[ 1 ] == 5318 ).toBeTruthy();
			expect( lhs.elements[ 2 ] == 10514 ).toBeTruthy();
			expect( lhs.elements[ 3 ] == 15894 ).toBeTruthy();
			expect( lhs.elements[ 4 ] == 1655 ).toBeTruthy();
			expect( lhs.elements[ 5 ] == 5562 ).toBeTruthy();
			expect( lhs.elements[ 6 ] == 11006 ).toBeTruthy();
			expect( lhs.elements[ 7 ] == 16634 ).toBeTruthy();
			expect( lhs.elements[ 8 ] == 1787 ).toBeTruthy();
			expect( lhs.elements[ 9 ] == 5980 ).toBeTruthy();
			expect( lhs.elements[ 10 ] == 11840 ).toBeTruthy();
			expect( lhs.elements[ 11 ] == 17888 ).toBeTruthy();
			expect( lhs.elements[ 12 ] == 1861 ).toBeTruthy();
			expect( lhs.elements[ 13 ] == 6246 ).toBeTruthy();
			expect( lhs.elements[ 14 ] == 12378 ).toBeTruthy();
			expect( lhs.elements[ 15 ] == 18710 ).toBeTruthy();

		} );

		test( 'premultiply', () => {

			const lhs = new Matrix4().set( 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 );
			const rhs = new Matrix4().set( 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131 );

			rhs.premultiply( lhs );

			expect( rhs.elements[ 0 ] == 1585 ).toBeTruthy();
			expect( rhs.elements[ 1 ] == 5318 ).toBeTruthy();
			expect( rhs.elements[ 2 ] == 10514 ).toBeTruthy();
			expect( rhs.elements[ 3 ] == 15894 ).toBeTruthy();
			expect( rhs.elements[ 4 ] == 1655 ).toBeTruthy();
			expect( rhs.elements[ 5 ] == 5562 ).toBeTruthy();
			expect( rhs.elements[ 6 ] == 11006 ).toBeTruthy();
			expect( rhs.elements[ 7 ] == 16634 ).toBeTruthy();
			expect( rhs.elements[ 8 ] == 1787 ).toBeTruthy();
			expect( rhs.elements[ 9 ] == 5980 ).toBeTruthy();
			expect( rhs.elements[ 10 ] == 11840 ).toBeTruthy();
			expect( rhs.elements[ 11 ] == 17888 ).toBeTruthy();
			expect( rhs.elements[ 12 ] == 1861 ).toBeTruthy();
			expect( rhs.elements[ 13 ] == 6246 ).toBeTruthy();
			expect( rhs.elements[ 14 ] == 12378 ).toBeTruthy();
			expect( rhs.elements[ 15 ] == 18710 ).toBeTruthy();

		} );

		test( 'multiplyMatrices', () => {

			// Reference:
			//
			// #!/usr/bin/env python
			// from __future__ import print_function
			// import numpy as np
			// print(
			//     np.dot(
			//         np.reshape([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53], (4, 4)),
			//         np.reshape([59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131], (4, 4))
			//     )
			// )
			//
			// [[ 1585  1655  1787  1861]
			//  [ 5318  5562  5980  6246]
			//  [10514 11006 11840 12378]
			//  [15894 16634 17888 18710]]
			const lhs = new Matrix4().set( 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 );
			const rhs = new Matrix4().set( 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131 );
			const ans = new Matrix4();

			ans.multiplyMatrices( lhs, rhs );

			expect( ans.elements[ 0 ] == 1585 ).toBeTruthy();
			expect( ans.elements[ 1 ] == 5318 ).toBeTruthy();
			expect( ans.elements[ 2 ] == 10514 ).toBeTruthy();
			expect( ans.elements[ 3 ] == 15894 ).toBeTruthy();
			expect( ans.elements[ 4 ] == 1655 ).toBeTruthy();
			expect( ans.elements[ 5 ] == 5562 ).toBeTruthy();
			expect( ans.elements[ 6 ] == 11006 ).toBeTruthy();
			expect( ans.elements[ 7 ] == 16634 ).toBeTruthy();
			expect( ans.elements[ 8 ] == 1787 ).toBeTruthy();
			expect( ans.elements[ 9 ] == 5980 ).toBeTruthy();
			expect( ans.elements[ 10 ] == 11840 ).toBeTruthy();
			expect( ans.elements[ 11 ] == 17888 ).toBeTruthy();
			expect( ans.elements[ 12 ] == 1861 ).toBeTruthy();
			expect( ans.elements[ 13 ] == 6246 ).toBeTruthy();
			expect( ans.elements[ 14 ] == 12378 ).toBeTruthy();
			expect( ans.elements[ 15 ] == 18710 ).toBeTruthy();

		} );

		test( 'multiplyScalar', () => {

			const b = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			expect( b.elements[ 0 ] == 0 ).toBeTruthy();
			expect( b.elements[ 1 ] == 4 ).toBeTruthy();
			expect( b.elements[ 2 ] == 8 ).toBeTruthy();
			expect( b.elements[ 3 ] == 12 ).toBeTruthy();
			expect( b.elements[ 4 ] == 1 ).toBeTruthy();
			expect( b.elements[ 5 ] == 5 ).toBeTruthy();
			expect( b.elements[ 6 ] == 9 ).toBeTruthy();
			expect( b.elements[ 7 ] == 13 ).toBeTruthy();
			expect( b.elements[ 8 ] == 2 ).toBeTruthy();
			expect( b.elements[ 9 ] == 6 ).toBeTruthy();
			expect( b.elements[ 10 ] == 10 ).toBeTruthy();
			expect( b.elements[ 11 ] == 14 ).toBeTruthy();
			expect( b.elements[ 12 ] == 3 ).toBeTruthy();
			expect( b.elements[ 13 ] == 7 ).toBeTruthy();
			expect( b.elements[ 14 ] == 11 ).toBeTruthy();
			expect( b.elements[ 15 ] == 15 ).toBeTruthy();

			b.multiplyScalar( 2 );
			expect( b.elements[ 0 ] == 0 * 2 ).toBeTruthy();
			expect( b.elements[ 1 ] == 4 * 2 ).toBeTruthy();
			expect( b.elements[ 2 ] == 8 * 2 ).toBeTruthy();
			expect( b.elements[ 3 ] == 12 * 2 ).toBeTruthy();
			expect( b.elements[ 4 ] == 1 * 2 ).toBeTruthy();
			expect( b.elements[ 5 ] == 5 * 2 ).toBeTruthy();
			expect( b.elements[ 6 ] == 9 * 2 ).toBeTruthy();
			expect( b.elements[ 7 ] == 13 * 2 ).toBeTruthy();
			expect( b.elements[ 8 ] == 2 * 2 ).toBeTruthy();
			expect( b.elements[ 9 ] == 6 * 2 ).toBeTruthy();
			expect( b.elements[ 10 ] == 10 * 2 ).toBeTruthy();
			expect( b.elements[ 11 ] == 14 * 2 ).toBeTruthy();
			expect( b.elements[ 12 ] == 3 * 2 ).toBeTruthy();
			expect( b.elements[ 13 ] == 7 * 2 ).toBeTruthy();
			expect( b.elements[ 14 ] == 11 * 2 ).toBeTruthy();
			expect( b.elements[ 15 ] == 15 * 2 ).toBeTruthy();

		} );

		test( 'determinant', () => {

			const a = new Matrix4();
			expect( a.determinant() == 1 ).toBeTruthy();

			a.elements[ 0 ] = 2;
			expect( a.determinant() == 2 ).toBeTruthy();

			a.elements[ 0 ] = 0;
			expect( a.determinant() == 0 ).toBeTruthy();

			// calculated via http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
			a.set( 2, 3, 4, 5, - 1, - 21, - 3, - 4, 6, 7, 8, 10, - 8, - 9, - 10, - 12 );
			expect( a.determinant() == 76 ).toBeTruthy();

		} );

		test( 'determinantAffine', () => {

			// for affine matrices (the typical object world matrix), the 3x3 result
			// equals the full 4x4 determinant since the bottom row is [ 0, 0, 0, 1 ]

			const a = new Matrix4();
			const position = new Vector3( 5, - 2, 3 );
			const quaternion = new Quaternion().setFromEuler( new Euler( 0.1, - 0.7, 1.3 ) );

			// translation + rotation + non-uniform scale

			a.compose( position, quaternion, new Vector3( 2, 3, 0.5 ) );
			expect( Math.abs( a.determinantAffine() - a.determinant() ) <= eps ).toBeTruthy();

			// reflection (negative scale on one axis flips the winding order)

			a.compose( position, quaternion, new Vector3( 2, 3, - 0.5 ) );
			expect( a.determinantAffine() < 0 ).toBeTruthy();
			expect( Math.abs( a.determinantAffine() - a.determinant() ) <= eps ).toBeTruthy();

			// shear

			a.multiply( new Matrix4().makeShear( 0.5, 0, 0.2, 0, 0.7, 0 ) );
			expect( Math.abs( a.determinantAffine() - a.determinant() ) <= eps ).toBeTruthy();

		} );

		test( 'determinantAffine (projective matrix)', () => {

			// for non-affine (projective) matrices, the bottom row is not [ 0, 0, 0, 1 ]
			// and so the 3x3 result generally differs from the full 4x4 determinant

			const a = new Matrix4().makePerspective( - 1, 1, 1, - 1, 1, 100 );
			expect( Math.abs( a.determinantAffine() - a.determinant() ) > eps ).toBeTruthy();

		} );

		test( 'transpose', () => {

			const a = new Matrix4();
			let b = a.clone().transpose();
			expect( matrixEquals4( a, b ) ).toBeTruthy();

			b = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			const c = b.clone().transpose();
			expect( ! matrixEquals4( b, c ) ).toBeTruthy();
			c.transpose();
			expect( matrixEquals4( b, c ) ).toBeTruthy();

		} );

		test( 'setPosition', () => {

			const a = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			const b = new Vector3( - 1, - 2, - 3 );
			const c = new Matrix4().set( 0, 1, 2, - 1, 4, 5, 6, - 2, 8, 9, 10, - 3, 12, 13, 14, 15 );

			a.setPosition( b );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

			const d = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
			const e = new Matrix4().set( 0, 1, 2, - 1, 4, 5, 6, - 2, 8, 9, 10, - 3, 12, 13, 14, 15 );

			d.setPosition( - 1, - 2, - 3 );
			expect( matrixEquals4( d, e ) ).toBeTruthy();

		} );

		test( 'invert', () => {

			const zero = new Matrix4().set( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
			const identity = new Matrix4();

			const a = new Matrix4();
			const b = new Matrix4().set( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );

			a.copy( b ).invert();
			expect( matrixEquals4( a, zero ) ).toBeTruthy();

			const testMatrices = [
				new Matrix4().makeRotationX( 0.3 ),
				new Matrix4().makeRotationX( - 0.3 ),
				new Matrix4().makeRotationY( 0.3 ),
				new Matrix4().makeRotationY( - 0.3 ),
				new Matrix4().makeRotationZ( 0.3 ),
				new Matrix4().makeRotationZ( - 0.3 ),
				new Matrix4().makeScale( 1, 2, 3 ),
				new Matrix4().makeScale( 1 / 8, 1 / 2, 1 / 3 ),
				new Matrix4().makePerspective( - 1, 1, 1, - 1, 1, 1000 ),
				new Matrix4().makePerspective( - 16, 16, 9, - 9, 0.1, 10000 ),
				new Matrix4().makeTranslation( 1, 2, 3 )
			];

			for ( let i = 0, il = testMatrices.length; i < il; i ++ ) {

				const m = testMatrices[ i ];

				const mInverse = new Matrix4().copy( m ).invert();
				const mSelfInverse = m.clone();
				mSelfInverse.copy( mSelfInverse ).invert();

				// self-inverse should the same as inverse
				expect( matrixEquals4( mSelfInverse, mInverse ) ).toBeTruthy();

				// the determinant of the inverse should be the reciprocal
				expect( Math.abs( m.determinant() * mInverse.determinant() - 1 ) < 0.0001 ).toBeTruthy();

				const mProduct = new Matrix4().multiplyMatrices( m, mInverse );

				// the determinant of the identity matrix is 1
				expect( Math.abs( mProduct.determinant() - 1 ) < 0.0001 ).toBeTruthy();
				expect( matrixEquals4( mProduct, identity ) ).toBeTruthy();

			}

		} );

		test( 'scale', () => {

			const a = new Matrix4().set( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
			const b = new Vector3( 2, 3, 4 );
			const c = new Matrix4().set( 2, 6, 12, 4, 10, 18, 28, 8, 18, 30, 44, 12, 26, 42, 60, 16 );

			a.scale( b );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'getMaxScaleOnAxis', () => {

			const a = new Matrix4().set( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
			const expected = Math.sqrt( 3 * 3 + 7 * 7 + 11 * 11 );

			expect( Math.abs( a.getMaxScaleOnAxis() - expected ) <= eps ).toBeTruthy();

		} );

		test( 'makeTranslation', () => {

			const a = new Matrix4();
			const b = new Vector3( 2, 3, 4 );
			const c = new Matrix4().set( 1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4, 0, 0, 0, 1 );

			a.makeTranslation( b.x, b.y, b.z );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

			a.makeTranslation( b );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'makeRotationX', () => {

			const a = new Matrix4();
			const b = Math.sqrt( 3 ) / 2;
			const c = new Matrix4().set( 1, 0, 0, 0, 0, b, - 0.5, 0, 0, 0.5, b, 0, 0, 0, 0, 1 );

			a.makeRotationX( Math.PI / 6 );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'makeRotationY', () => {

			const a = new Matrix4();
			const b = Math.sqrt( 3 ) / 2;
			const c = new Matrix4().set( b, 0, 0.5, 0, 0, 1, 0, 0, - 0.5, 0, b, 0, 0, 0, 0, 1 );

			a.makeRotationY( Math.PI / 6 );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'makeRotationZ', () => {

			const a = new Matrix4();
			const b = Math.sqrt( 3 ) / 2;
			const c = new Matrix4().set( b, - 0.5, 0, 0, 0.5, b, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 );

			a.makeRotationZ( Math.PI / 6 );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'makeRotationAxis', () => {

			const axis = new Vector3( 1.5, 0.0, 1.0 ).normalize();
			const radians = MathUtils.degToRad( 45 );
			const a = new Matrix4().makeRotationAxis( axis, radians );

			const expected = new Matrix4().set(
				0.9098790095958609, - 0.39223227027636803, 0.13518148560620882, 0,
				0.39223227027636803, 0.7071067811865476, - 0.588348405414552, 0,
				0.13518148560620882, 0.588348405414552, 0.7972277715906868, 0,
				0, 0, 0, 1
			);

			expect( matrixEquals4( a, expected ) ).toBeTruthy();

		} );

		test( 'makeScale', () => {

			const a = new Matrix4();
			const c = new Matrix4().set( 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1 );

			a.makeScale( 2, 3, 4 );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'makeShear', () => {

			const a = new Matrix4();
			const c = new Matrix4().set( 1, 3, 5, 0, 1, 1, 6, 0, 2, 4, 1, 0, 0, 0, 0, 1 );

			a.makeShear( 1, 2, 3, 4, 5, 6 );
			expect( matrixEquals4( a, c ) ).toBeTruthy();

		} );

		test( 'compose/decompose', () => {

			const tValues = [
				new Vector3(),
				new Vector3( 3, 0, 0 ),
				new Vector3( 0, 4, 0 ),
				new Vector3( 0, 0, 5 ),
				new Vector3( - 6, 0, 0 ),
				new Vector3( 0, - 7, 0 ),
				new Vector3( 0, 0, - 8 ),
				new Vector3( - 2, 5, - 9 ),
				new Vector3( - 2, - 5, - 9 )
			];

			const sValues = [
				new Vector3( 1, 1, 1 ),
				new Vector3( 2, 2, 2 ),
				new Vector3( 1, - 1, 1 ),
				new Vector3( - 1, 1, 1 ),
				new Vector3( 1, 1, - 1 ),
				new Vector3( 2, - 2, 1 ),
				new Vector3( - 1, 2, - 2 ),
				new Vector3( - 1, - 1, - 1 ),
				new Vector3( - 2, - 2, - 2 )
			];

			const rValues = [
				new Quaternion(),
				new Quaternion().setFromEuler( new Euler( 1, 1, 0 ) ),
				new Quaternion().setFromEuler( new Euler( 1, - 1, 1 ) ),
				new Quaternion( 0, 0.9238795292366128, 0, 0.38268342717215614 )
			];

			for ( let ti = 0; ti < tValues.length; ti ++ ) {

				for ( let si = 0; si < sValues.length; si ++ ) {

					for ( let ri = 0; ri < rValues.length; ri ++ ) {

						const t = tValues[ ti ];
						const s = sValues[ si ];
						const r = rValues[ ri ];

						const m = new Matrix4().compose( t, r, s );
						const t2 = new Vector3();
						const r2 = new Quaternion();
						const s2 = new Vector3();

						m.decompose( t2, r2, s2 );

						const m2 = new Matrix4().compose( t2, r2, s2 );

						/*
						// debug code
						const matrixIsSame = matrixEquals4( m, m2 );
						if ( ! matrixIsSame ) {

							console.log( t, s, r );
							console.log( t2, s2, r2 );
							console.log( m, m2 );

						}
						*/

						expect( matrixEquals4( m, m2 ) ).toBeTruthy();

					}

				}

			}

		} );

		test( 'makePerspective', () => {

			const a = new Matrix4().makePerspective( - 1, 1, - 1, 1, 1, 100 );
			const expected = new Matrix4().set(
				1, 0, 0, 0,
				0, - 1, 0, 0,
				0, 0, - 101 / 99, - 200 / 99,
				0, 0, - 1, 0
			);
			expect( matrixEquals4( a, expected ) ).toBeTruthy();

		} );

		test( 'makeOrthographic', () => {

			const a = new Matrix4().makeOrthographic( - 1, 1, - 1, 1, 1, 100 );
			const expected = new Matrix4().set(
				1, 0, 0, 0,
				0, - 1, 0, 0,
				0, 0, - 2 / 99, - 101 / 99,
				0, 0, 0, 1
			);

			expect( matrixEquals4( a, expected ) ).toBeTruthy();

		} );

		test( 'equals', () => {

			const a = new Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
			const b = new Matrix4().set( 0, - 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );

			expect( a.equals( b ) ).toBeFalsy();
			expect( b.equals( a ) ).toBeFalsy();

			a.copy( b );
			expect( a.equals( b ) ).toBeTruthy();
			expect( b.equals( a ) ).toBeTruthy();

		} );

		test( 'fromArray', () => {

			const a = new Matrix4();
			const b = new Matrix4().set( 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 );

			a.fromArray( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ] );
			expect( a.equals( b ) ).toBeTruthy();

		} );

		test( 'toArray', () => {

			const a = new Matrix4().set( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
			const noOffset = [ 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 ];
			const withOffset = [ undefined, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16 ];

			let array = a.toArray();
			expect( array ).toEqual( noOffset );

			array = [];
			a.toArray( array );
			expect( array ).toEqual( noOffset );

			array = [];
			a.toArray( array, 1 );
			expect( array ).toEqual( withOffset );

		} );

	} );

} );

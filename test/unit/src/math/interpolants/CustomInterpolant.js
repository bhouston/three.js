import { describe, test, expect } from 'vitest';
import { Interpolant } from '@src/math/Interpolant.js';

describe( 'Maths', () => {

	describe( 'Interpolants', () => {

		describe( 'CustomInterpolant', () => {

			// A custom cubic spline interpolant mimicking `GLTFCubicSplineInterpolant`
			// from `GLTFLoader`. The keyframe layout for CUBICSPLINE animations is:
			// [ inTangent_1, splineVertex_1, outTangent_1, inTangent_2, splineVertex_2, ... ]

			class CubicSplineInterpolant extends Interpolant {

				constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {

					super( parameterPositions, sampleValues, sampleSize, resultBuffer );

				}

				copySampleValue_( index ) {

					const result = this.resultBuffer,
						values = this.sampleValues,
						valueSize = this.valueSize,
						offset = index * valueSize * 3 + valueSize;

					for ( let i = 0; i !== valueSize; i ++ ) {

						result[ i ] = values[ offset + i ];

					}

					return result;

				}

				interpolate_( i1, t0, t, t1 ) {

					const result = this.resultBuffer;
					const values = this.sampleValues;
					const stride = this.valueSize;

					const stride2 = stride * 2;
					const stride3 = stride * 3;

					const td = t1 - t0;

					const p = ( t - t0 ) / td;
					const pp = p * p;
					const ppp = pp * p;

					const offset1 = i1 * stride3;
					const offset0 = offset1 - stride3;

					const s2 = - 2 * ppp + 3 * pp;
					const s3 = ppp - pp;
					const s0 = 1 - s2;
					const s1 = s3 - pp + p;

					for ( let i = 0; i !== stride; i ++ ) {

						const p0 = values[ offset0 + i + stride ];
						const m0 = values[ offset0 + i + stride2 ] * td;
						const p1 = values[ offset1 + i + stride ];
						const m1 = values[ offset1 + i ] * td;

						result[ i ] = s0 * p0 + s1 * m0 + s2 * p1 + s3 * m1;

					}

					return result;

				}

			}

			test( 'Extending', () => {

				// parameterPositions, sampleValues, sampleSize, resultBuffer
				const object = new CubicSplineInterpolant( [ 0, 1 ], [ 0, 0, 0, 0, 0, 0 ], 1, [] );
				expect( object instanceof Interpolant ).toBe( true );

			} );

			test( 'evaluate', () => {

				// Two keyframes at t = 0 and t = 1, valueSize = 1.
				// Layout: [ in_0, v_0, out_0, in_1, v_1, out_1 ]
				// Vertex values 0 -> 1 with non-zero tangents to exercise all spline terms.
				const positions = [ 0, 1 ];
				const values = [ 0, 0, 1, - 1, 1, 0 ];
				const interpolant = new CubicSplineInterpolant( positions, values, 1, [ 0 ] );

				expect( interpolant.evaluate( 0 ) ).toEqual( [ 0 ] );
				expect( interpolant.evaluate( 1 ) ).toEqual( [ 1 ] );

				// At t = 0.5 with td = 1, p = 0.5 → s0 = 0.5, s1 = 0.125, s2 = 0.5, s3 = -0.125
				// result = 0.5 * 0 + 0.125 * 1 + 0.5 * 1 + ( -0.125 ) * ( -1 ) = 0.75
				expect( interpolant.evaluate( 0.5 ) ).toEqual( [ 0.75 ] );

				// Out-of-range queries clamp to the boundary spline vertex.
				expect( interpolant.evaluate( - 1 ) ).toEqual( [ 0 ] );
				expect( interpolant.evaluate( 2 ) ).toEqual( [ 1 ] );

			} );

		} );

	} );

} );

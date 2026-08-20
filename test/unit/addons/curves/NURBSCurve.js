import { describe, test, expect, beforeAll } from 'vitest';
import { NURBSCurve } from '../../../../examples/jsm/curves/NURBSCurve.js';
import { MathUtils, Vector4 } from 'three';

describe( 'Extras', () => {

	describe( 'Curves', () => {

		describe( 'NURBSCurve', () => {

			let _nurbsCurve = undefined;

			beforeAll( () => {

				const nurbsControlPoints = [];
				const nurbsKnots = [];
				const nurbsDegree = 3;

				for ( let i = 0; i <= nurbsDegree; i ++ ) {

					nurbsKnots.push( 0 );

				}

				for ( let i = 0, j = 20; i < j; i ++ ) {

					const point = new Vector4( Math.random(), Math.random(), Math.random(), 1 );
					nurbsControlPoints.push( point );

					const knot = ( i + 1 ) / ( j - nurbsDegree );
					nurbsKnots.push( MathUtils.clamp( knot, 0, 1 ) );

				}

				_nurbsCurve = new NURBSCurve( nurbsDegree, nurbsKnots, nurbsControlPoints );

			} );

			test( 'toJSON', () => {

				const json = _nurbsCurve.toJSON();

				expect( json.degree ).toBe( _nurbsCurve.degree );
				expect( json.knots ).toEqual( _nurbsCurve.knots );
				expect( json.controlPoints ).toEqual( _nurbsCurve.controlPoints.map( p => p.toArray() ) );
				expect( json.startKnot ).toBe( _nurbsCurve.startKnot );
				expect( json.endKnot ).toBe( _nurbsCurve.endKnot );

			} );

			test( 'fromJSON', () => {

				const json = _nurbsCurve.toJSON();
				const fromJson = new NURBSCurve().fromJSON( json );

				expect( fromJson.degree ).toBe( _nurbsCurve.degree );
				expect( fromJson.knots ).toEqual( _nurbsCurve.knots );
				expect( fromJson.controlPoints ).toEqual( _nurbsCurve.controlPoints );
				expect( fromJson.startKnot ).toBe( _nurbsCurve.startKnot );
				expect( fromJson.endKnot ).toBe( _nurbsCurve.endKnot );

			} );

		} );

	} );

} );

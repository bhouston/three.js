import { describe, test, expect, beforeAll } from 'vitest';
import { CubicBezierCurve } from '@src/extras/curves/CubicBezierCurve.js';

import { Curve } from '@src/extras/core/Curve.js';
import { Vector2 } from '@src/math/Vector2.js';

describe( 'Extras', () => {

	describe( 'Curves', () => {

		describe( 'CubicBezierCurve', () => {

			let curve = undefined;
			beforeAll( () => {

				curve = new CubicBezierCurve(
					new Vector2( - 10, 0 ),
					new Vector2( - 5, 15 ),
					new Vector2( 20, 15 ),
					new Vector2( 10, 0 )
				);

			} );

			test( 'Extending', () => {

				const object = new CubicBezierCurve();
				expect( object instanceof Curve ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new CubicBezierCurve();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new CubicBezierCurve();
				expect( object.type === 'CubicBezierCurve' ).toBeTruthy();

			} );

			test( 'isCubicBezierCurve', () => {

				const object = new CubicBezierCurve();
				expect( object.isCubicBezierCurve ).toBeTruthy();

			} );

			test( 'Simple curve', () => {

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( - 3.359375, 8.4375 ),
					new Vector2( 5.625, 11.25 ),
					new Vector2( 11.796875, 8.4375 ),
					new Vector2( 10, 0 )
				];

				let points = curve.getPoints( expectedPoints.length - 1 );

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints );

				// symmetry
				const curveRev = new CubicBezierCurve(
					curve.v3, curve.v2, curve.v1, curve.v0
				);

				points = curveRev.getPoints( expectedPoints.length - 1 );

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints.reverse() );

			} );

			test( 'getLength/getLengths', () => {

				const length = curve.getLength();
				const expectedLength = 36.64630888504102;

				expect( length ).toNumEqual( expectedLength );

				const expectedLengths = [
					0,
					10.737285813492393,
					20.15159143794633,
					26.93408340370825,
					35.56079575637337
				];
				const lengths = curve.getLengths( expectedLengths.length - 1 );

				expect( lengths.length ).toBe( expectedLengths.length );

				lengths.forEach( function ( segment, i ) {

					expect( segment ).toNumEqual( expectedLengths[ i ] );

				} );

			} );

			test( 'getPointAt', () => {

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( - 3.3188282598022596, 8.463722639089221 ),
					new Vector2( 3.4718554735926617, 11.07899406116314 ),
					new Vector2( 10, 0 )
				];

				const points = [
					curve.getPointAt( 0, new Vector2() ),
					curve.getPointAt( 0.3, new Vector2() ),
					curve.getPointAt( 0.5, new Vector2() ),
					curve.getPointAt( 1, new Vector2() )
				];

				expect( points ).toEqual( expectedPoints );

			} );

			test( 'getTangent/getTangentAt', () => {

				let expectedTangents = [
					new Vector2( 0.316370061632252, 0.9486358543207215 ),
					new Vector2( 0.838961283088303, 0.5441911111721949 ),
					new Vector2( 1, 0 ),
					new Vector2( 0.47628313192245453, - 0.8792919755383518 ),
					new Vector2( - 0.5546041767829665, - 0.8321142992972107 )
				];

				let tangents = [
					curve.getTangent( 0, new Vector2() ),
					curve.getTangent( 0.25, new Vector2() ),
					curve.getTangent( 0.5, new Vector2() ),
					curve.getTangent( 0.75, new Vector2() ),
					curve.getTangent( 1, new Vector2() )
				];

				expectedTangents.forEach( function ( exp, i ) {

					const tangent = tangents[ i ];

					expect( tangent.x ).toNumEqual( exp.x );
					expect( tangent.y ).toNumEqual( exp.y );

				} );

				//

				expectedTangents = [
					new Vector2( 0.316370061632252, 0.9486358543207215 ),
					new Vector2( 0.7794223085548987, 0.6264988945935596 ),
					new Vector2( 0.988266153082452, 0.15274164681452052 ),
					new Vector2( 0.5004110404199416, - 0.8657879593906534 ),
					new Vector2( - 0.5546041767829665, - 0.8321142992972107 )
				];

				tangents = [
					curve.getTangentAt( 0, new Vector2() ),
					curve.getTangentAt( 0.25, new Vector2() ),
					curve.getTangentAt( 0.5, new Vector2() ),
					curve.getTangentAt( 0.75, new Vector2() ),
					curve.getTangentAt( 1, new Vector2() )
				];

				expectedTangents.forEach( function ( exp, i ) {

					const tangent = tangents[ i ];

					expect( tangent.x ).toNumEqual( exp.x );
					expect( tangent.y ).toNumEqual( exp.y );

				} );

			} );

			test( 'getUtoTmapping', () => {

				const start = curve.getUtoTmapping( 0, 0 );
				const end = curve.getUtoTmapping( 0, curve.getLength() );
				const somewhere = curve.getUtoTmapping( 0.5, 1 );

				const expectedSomewhere = 0.02130029182257093;

				expect( start ).toBe( 0 );
				expect( end ).toBe( 1 );
				expect( somewhere ).toNumEqual( expectedSomewhere );

			} );

			test( 'getSpacedPoints', () => {

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( - 6.16826457740703, 6.17025727295411 ),
					new Vector2( - 0.058874033259857184, 10.1240558653185 ),
					new Vector2( 7.123523032625162, 11.154913869041575 ),
					new Vector2( 12.301846885754463, 6.808865855469985 ),
					new Vector2( 10, 0 )
				];

				const points = curve.getSpacedPoints();

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints );

			} );

		} );

	} );

} );

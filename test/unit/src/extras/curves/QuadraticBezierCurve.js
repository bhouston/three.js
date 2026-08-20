import { describe, test, expect, beforeAll } from 'vitest';
import { QuadraticBezierCurve } from '@src/extras/curves/QuadraticBezierCurve.js';

import { Curve } from '@src/extras/core/Curve.js';
import { Vector2 } from '@src/math/Vector2.js';

describe( 'Extras', () => {

	describe( 'Curves', () => {

		describe( 'QuadraticBezierCurve', () => {

			let _curve = undefined;
			beforeAll( () => {

				_curve = new QuadraticBezierCurve(
					new Vector2( - 10, 0 ),
					new Vector2( 20, 15 ),
					new Vector2( 10, 0 )
				);

			} );

			test( 'Extending', () => {

				const object = new QuadraticBezierCurve();
				expect( object instanceof Curve ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new QuadraticBezierCurve();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new QuadraticBezierCurve();
				expect( object.type === 'QuadraticBezierCurve' ).toBeTruthy();

			} );

			test( 'isQuadraticBezierCurve', () => {

				const object = new QuadraticBezierCurve();
				expect( object.isQuadraticBezierCurve ).toBeTruthy();

			} );

			test( 'Simple curve', () => {

				const curve = _curve;

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( 2.5, 5.625 ),
					new Vector2( 10, 7.5 ),
					new Vector2( 12.5, 5.625 ),
					new Vector2( 10, 0 )
				];

				let points = curve.getPoints( expectedPoints.length - 1 );

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints );

				// symmetry
				const curveRev = new QuadraticBezierCurve(
					curve.v2, curve.v1, curve.v0
				);

				points = curveRev.getPoints( expectedPoints.length - 1 );

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints.reverse() );

			} );

			test( 'getLength/getLengths', () => {

				const curve = _curve;

				const length = curve.getLength();
				const expectedLength = 31.269026549416683;

				expect( length ).toNumEqual( expectedLength );

				const expectedLengths = [
					0,
					13.707320124663317,
					21.43814317269643,
					24.56314317269643,
					30.718679298818998
				];
				const lengths = curve.getLengths( expectedLengths.length - 1 );

				expect( lengths.length ).toBe( expectedLengths.length );

				lengths.forEach( function ( segment, i ) {

					expect( segment ).toNumEqual( expectedLengths[ i ] );

				} );

			} );

			test( 'getPointAt', () => {

				const curve = _curve;

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( - 1.5127849599387615, 3.993582003773624 ),
					new Vector2( 4.310076165722796, 6.269921971403917 ),
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

				const curve = _curve;

				let expectedTangents = [
					new Vector2( 0.89443315420562, 0.44720166888975904 ),
					new Vector2( 0.936329177569021, 0.3511234415884543 ),
					new Vector2( 1, 0 ),
					new Vector2( - 5.921189464667277e-13, - 1 ),
					new Vector2( - 0.5546617882904897, - 0.8320758983472577 )
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
					new Vector2( 0.89443315420562, 0.44720166888975904 ),
					new Vector2( 0.9125211423360805, 0.40902954024086674 ),
					new Vector2( 0.9480289098765387, 0.3181842014278863 ),
					new Vector2( 0.7969127189169473, - 0.6040944615111106 ),
					new Vector2( - 0.5546617882904897, - 0.8320758983472577 )
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

				const curve = _curve;

				const start = curve.getUtoTmapping( 0, 0 );
				const end = curve.getUtoTmapping( 0, curve.getLength() );
				const somewhere = curve.getUtoTmapping( 0.5, 1 );

				const expectedSomewhere = 0.015073978276116116;

				expect( start ).toBe( 0 );
				expect( end ).toBe( 1 );
				expect( somewhere ).toNumEqual( expectedSomewhere );

			} );

			test( 'getSpacedPoints', () => {

				const curve = _curve;

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( - 4.366603655406173, 2.715408933540383 ),
					new Vector2( 1.3752241477827831, 5.191972084404416 ),
					new Vector2( 7.312990279153634, 7.136310044848586 ),
					new Vector2( 12.499856644824826, 5.653289188715387 ),
					new Vector2( 10, 0 )
				];

				const points = curve.getSpacedPoints();

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints );

			} );

		} );

	} );

} );

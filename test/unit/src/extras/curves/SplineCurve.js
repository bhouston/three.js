import { describe, test, expect, beforeAll } from 'vitest';
import { SplineCurve } from '@src/extras/curves/SplineCurve.js';

import { Curve } from '@src/extras/core/Curve.js';
import { Vector2 } from '@src/math/Vector2.js';

describe( 'Extras', () => {

	describe( 'Curves', () => {

		describe( 'SplineCurve', () => {

			let _curve = undefined;
			beforeAll( () => {

				_curve = new SplineCurve( [
					new Vector2( - 10, 0 ),
					new Vector2( - 5, 5 ),
					new Vector2( 0, 0 ),
					new Vector2( 5, - 5 ),
					new Vector2( 10, 0 )
				] );

			} );

			test( 'Extending', () => {

				const object = new SplineCurve();
				expect( object instanceof Curve ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new SplineCurve();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new SplineCurve();
				expect( object.type === 'SplineCurve' ).toBeTruthy();

			} );

			test( 'isSplineCurve', () => {

				const object = new SplineCurve();
				expect( object.isSplineCurve ).toBeTruthy();

			} );

			test( 'Simple curve', () => {

				const curve = _curve;

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( - 6.08, 4.56 ),
					new Vector2( - 2, 2.48 ),
					new Vector2( 2, - 2.48 ),
					new Vector2( 6.08, - 4.56 ),
					new Vector2( 10, 0 )
				];

				let points = curve.getPoints( 5 );

				expect( points.length ).toBe( expectedPoints.length );

				points.forEach( function ( point, i ) {

					expect( point.x ).toNumEqual( expectedPoints[ i ].x );
					expect( point.y ).toNumEqual( expectedPoints[ i ].y );

				} );

				//

				points = curve.getPoints( 4 );

				expect( points ).toEqual( curve.points );

			} );

			test( 'getLength/getLengths', () => {

				const curve = _curve;

				const length = curve.getLength();
				const expectedLength = 28.876950901868135;

				expect( length ).toNumEqual( expectedLength );

				const expectedLengths = [
					0.0,
					Math.sqrt( 50 ),
					Math.sqrt( 200 ),
					Math.sqrt( 450 ),
					Math.sqrt( 800 )
				];

				const lengths = curve.getLengths( 4 );

				expect( lengths ).toEqual( expectedLengths );

			} );

			test( 'getPointAt', () => {

				const curve = _curve;
				const point = new Vector2();

				expect( curve.getPointAt( 0, point ).equals( curve.points[ 0 ] ) ).toBeTruthy();
				expect( curve.getPointAt( 1, point ).equals( curve.points[ 4 ] ) ).toBeTruthy();

				curve.getPointAt( 0.5, point );

				expect( point.x ).toNumEqual( 0.0 );
				expect( point.y ).toNumEqual( 0.0 );

			} );

			test( 'getTangent', () => {

				const curve = _curve;

				const expectedTangent = [
					new Vector2( 0.7068243340243188, 0.7073891155729485 ), // 0
					new Vector2( 0.7069654305325396, - 0.7072481035902046 ), // 0.5
					new Vector2( 0.7068243340245123, 0.7073891155727552 ) // 1
				];

				const tangents = [
					curve.getTangent( 0, new Vector2() ),
					curve.getTangent( 0.5, new Vector2() ),
					curve.getTangent( 1, new Vector2() )
				];

				tangents.forEach( function ( tangent, i ) {

					expect( tangent.x ).toNumEqual( expectedTangent[ i ].x );
					expect( tangent.y ).toNumEqual( expectedTangent[ i ].y );

				} );

			} );

			test( 'getUtoTmapping', () => {

				const curve = _curve;

				const start = curve.getUtoTmapping( 0, 0 );
				const end = curve.getUtoTmapping( 0, curve.getLength() );
				const middle = curve.getUtoTmapping( 0.5, 0 );

				expect( start ).toBe( 0 );
				expect( end ).toBe( 1 );
				expect( middle ).toNumEqual( 0.5 );

			} );

			test( 'getSpacedPoints', () => {

				const curve = _curve;

				const expectedPoints = [
					new Vector2( - 10, 0 ),
					new Vector2( - 4.996509634683014, 4.999995128640857 ),
					new Vector2( 0, 0 ),
					new Vector2( 4.996509634683006, - 4.999995128640857 ),
					new Vector2( 10, 0 )
				];

				const points = curve.getSpacedPoints( 4 );

				expect( points.length ).toBe( expectedPoints.length );

				points.forEach( function ( point, i ) {

					expect( point.x ).toNumEqual( expectedPoints[ i ].x );
					expect( point.y ).toNumEqual( expectedPoints[ i ].y );

				} );

			} );

		} );

	} );

} );

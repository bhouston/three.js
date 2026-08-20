import { describe, test, expect, beforeAll } from 'vitest';
import { EllipseCurve } from '@src/extras/curves/EllipseCurve.js';

import { Curve } from '@src/extras/core/Curve.js';
import { Vector2 } from '@src/math/Vector2.js';

describe( 'Extras', () => {

	describe( 'Curves', () => {

		describe( 'EllipseCurve', () => {

			let curve = undefined;
			beforeAll( () => {

				curve = new EllipseCurve(
					0, 0, // ax, aY
					10, 10, // xRadius, yRadius
					0, 2 * Math.PI, // aStartAngle, aEndAngle
					false, // aClockwise
					0 // aRotation
				);

			} );

			test( 'Extending', () => {

				const object = new EllipseCurve();
				expect( object instanceof Curve ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new EllipseCurve();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new EllipseCurve();
				expect( object.type === 'EllipseCurve' ).toBeTruthy();

			} );

			test( 'isEllipseCurve', () => {

				const object = new EllipseCurve();
				expect( object.isEllipseCurve ).toBeTruthy();

			} );

			test( 'Simple curve', () => {

				const expectedPoints = [
					new Vector2( 10, 0 ),
					new Vector2( 0, 10 ),
					new Vector2( - 10, 0 ),
					new Vector2( 0, - 10 ),
					new Vector2( 10, 0 )
				];

				const points = curve.getPoints( expectedPoints.length - 1 );

				expect( points.length ).toBe( expectedPoints.length );

				points.forEach( function ( point, i ) {

					expect( point.x ).toNumEqual( expectedPoints[ i ].x );
					expect( point.y ).toNumEqual( expectedPoints[ i ].y );

				} );

			} );

			test( 'getLength/getLengths', () => {

				const length = curve.getLength();
				const expectedLength = 62.829269247282795;

				expect( length ).toNumEqual( expectedLength );

				const lengths = curve.getLengths( 5 );
				const expectedLengths = [
					0,
					11.755705045849462,
					23.51141009169892,
					35.26711513754839,
					47.02282018339785,
					58.77852522924731
				];

				expect( lengths.length ).toBe( expectedLengths.length );

				lengths.forEach( function ( segment, i ) {

					expect( segment ).toNumEqual( expectedLengths[ i ] );

				} );

			} );

			test( 'getPoint/getPointAt', () => {

				const testValues = [ 0, 0.3, 0.5, 0.7, 1 ];

				const p = new Vector2();
				const a = new Vector2();

				testValues.forEach( function ( val ) {

					const expectedX = Math.cos( val * Math.PI * 2 ) * 10;
					const expectedY = Math.sin( val * Math.PI * 2 ) * 10;

					curve.getPoint( val, p );
					curve.getPointAt( val, a );

					expect( p.x ).toNumEqual( expectedX );
					expect( p.y ).toNumEqual( expectedY );

					expect( a.x ).toNumEqual( expectedX );
					expect( a.y ).toNumEqual( expectedY );

				} );

			} );

			test( 'getTangent', () => {

				const expectedTangents = [
					new Vector2( - 0.000314159260186071, 0.9999999506519786 ),
					new Vector2( - 1, 0 ),
					new Vector2( 0, - 1 ),
					new Vector2( 1, 0 ),
					new Vector2( 0.00031415926018600165, 0.9999999506519784 )
				];

				const tangents = [
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

			} );

			test( 'getUtoTmapping', () => {

				const start = curve.getUtoTmapping( 0, 0 );
				const end = curve.getUtoTmapping( 0, curve.getLength() );
				const somewhere = curve.getUtoTmapping( 0.7, 1 );

				const expectedSomewhere = 0.01591614882650014;

				expect( start ).toBe( 0 );
				expect( end ).toBe( 1 );
				expect( somewhere ).toNumEqual( expectedSomewhere );

			} );

			test( 'getSpacedPoints', () => {

				const expectedPoints = [
					new Vector2( 10, 0 ),
					new Vector2( 3.0901699437494603, 9.51056516295154 ),
					new Vector2( - 8.090169943749492, 5.877852522924707 ),
					new Vector2( - 8.090169943749459, - 5.877852522924751 ),
					new Vector2( 3.0901699437494807, - 9.510565162951533 ),
					new Vector2( 10, - 2.4492935982947065e-15 )
				];

				const points = curve.getSpacedPoints();

				expect( points.length ).toBe( expectedPoints.length );

				expectedPoints.forEach( function ( exp, i ) {

					const point = points[ i ];

					expect( point.x ).toNumEqual( exp.x );
					expect( point.y ).toNumEqual( exp.y );

				} );

			} );

		} );

	} );

} );

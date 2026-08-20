import { describe, test, expect } from 'vitest';
import { CatmullRomCurve3 } from '@src/extras/curves/CatmullRomCurve3.js';

import { Curve } from '@src/extras/core/Curve.js';
import { Vector3 } from '@src/math/Vector3.js';

describe( 'Extras', () => {

	describe( 'Curves', () => {

		describe( 'CatmullRomCurve3', () => {

			/* eslint-disable */
			const positions = [
				new Vector3( - 60, - 100,   60 ),
				new Vector3( - 60,    20,   60 ),
				new Vector3( - 60,   120,   60 ),
				new Vector3(   60,    20, - 60 ),
				new Vector3(   60, - 100, - 60 )
			];
			/* eslint-enable */

			test( 'Extending', () => {

				const object = new CatmullRomCurve3();
				expect( object instanceof Curve ).toBe( true );

			} );

			test( 'Instancing', () => {

				const object = new CatmullRomCurve3();
				expect( object ).toBeTruthy();

			} );

			test( 'type', () => {

				const object = new CatmullRomCurve3();
				expect( object.type === 'CatmullRomCurve3' ).toBeTruthy();

			} );

			test( 'isCatmullRomCurve3', () => {

				const object = new CatmullRomCurve3();
				expect( object.isCatmullRomCurve3 ).toBeTruthy();

			} );

			test( 'catmullrom check', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';

				const expectedPoints = [

					new Vector3( - 60, - 100, 60 ),
					new Vector3( - 60, - 51.04, 60 ),
					new Vector3( - 60, - 2.7199999999999998, 60 ),
					new Vector3( - 61.92, 44.48, 61.92 ),
					new Vector3( - 68.64, 95.36000000000001, 68.64 ),
					new Vector3( - 60, 120, 60 ),
					new Vector3( - 14.880000000000017, 95.36000000000001, 14.880000000000017 ),
					new Vector3( 41.75999999999997, 44.48000000000003, - 41.75999999999997 ),
					new Vector3( 67.68, - 2.720000000000023, - 67.68 ),
					new Vector3( 65.75999999999999, - 51.04000000000001, - 65.75999999999999 ),
					new Vector3( 60, - 100, - 60 )

				];

				const points = curve.getPoints( 10 );

				expect( points.length ).toBe( expectedPoints.length );

				points.forEach( function ( point, i ) {

					expect( point.x ).toNumEqual( expectedPoints[ i ].x );
					expect( point.y ).toNumEqual( expectedPoints[ i ].y );
					expect( point.z ).toNumEqual( expectedPoints[ i ].z );

				} );

			} );

			test( 'chordal basic check', () => {

				const curve = new CatmullRomCurve3( positions );

				curve.curveType = 'chordal';

				const expectedPoints = [
					new Vector3( - 60, - 100, 60 ),
					new Vector3( - 60, - 52, 60 ),
					new Vector3( - 60, - 4, 60 ),
					new Vector3( - 60.656435889910924, 41.62455386421379, 60.656435889910924 ),
					new Vector3( - 62.95396150459915, 87.31049238896205, 62.95396150459915 ),
					new Vector3( - 60, 120, 60 ),
					new Vector3( - 16.302568199486444, 114.1500463116312, 16.302568199486444 ),
					new Vector3( 42.998098664956586, 54.017050116427455, - 42.998098664956586 ),
					new Vector3( 63.542500175682434, - 1.137153397546383, - 63.542500175682434 ),
					new Vector3( 62.65687513176183, - 49.85286504815978, - 62.65687513176183 ),
					new Vector3( 60.00000000000001, - 100, - 60.00000000000001 )
				];

				const points = curve.getPoints( 10 );

				expect( points.length ).toBe( expectedPoints.length );

				points.forEach( function ( point, i ) {

					expect( point.x ).toNumEqual( expectedPoints[ i ].x );
					expect( point.y ).toNumEqual( expectedPoints[ i ].y );
					expect( point.z ).toNumEqual( expectedPoints[ i ].z );

				} );

			} );

			test( 'centripetal basic check', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'centripetal';

				const expectedPoints = [
					new Vector3( - 60, - 100, 60 ),
					new Vector3( - 60, - 51.47527724919028, 60 ),
					new Vector3( - 60, - 3.300369665587032, 60 ),
					new Vector3( - 61.13836565863938, 42.86306307781241, 61.13836565863938 ),
					new Vector3( - 65.1226454638772, 90.69743905511538, 65.1226454638772 ),
					new Vector3( - 60, 120, 60 ),
					new Vector3( - 15.620412575504497, 103.10790870179872, 15.620412575504497 ),
					new Vector3( 42.384384731047874, 48.35477686933143, - 42.384384731047874 ),
					new Vector3( 65.25545512241153, - 1.646250966068339, - 65.25545512241153 ),
					new Vector3( 63.94159134180865, - 50.234688224551256, - 63.94159134180865 ),
					new Vector3( 59.99999999999999, - 100, - 59.99999999999999 ),
				];

				const points = curve.getPoints( 10 );

				expect( points.length ).toBe( expectedPoints.length );

				points.forEach( function ( point, i ) {

					expect( point.x ).toNumEqual( expectedPoints[ i ].x );
					expect( point.y ).toNumEqual( expectedPoints[ i ].y );
					expect( point.z ).toNumEqual( expectedPoints[ i ].z );

				} );

			} );

			test( 'closed catmullrom basic check', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';
				curve.closed = true;

				const expectedPoints = [
					new Vector3( - 60, - 100, 60 ),
					new Vector3( - 67.5, - 46.25, 67.5 ),
					new Vector3( - 60, 20, 60 ),
					new Vector3( - 67.5, 83.75, 67.5 ),
					new Vector3( - 60, 120, 60 ),
					new Vector3( 0, 83.75, 0 ),
					new Vector3( 60, 20, - 60 ),
					new Vector3( 75, - 46.25, - 75 ),
					new Vector3( 60, - 100, - 60 ),
					new Vector3( 0, - 115, 0 ),
					new Vector3( - 60, - 100, 60 ),
				];

				const points = curve.getPoints( 10 );

				expect( points.length ).toBe( expectedPoints.length );

				points.forEach( function ( point, i ) {

					expect( point.x ).toNumEqual( expectedPoints[ i ].x );
					expect( point.y ).toNumEqual( expectedPoints[ i ].y );
					expect( point.z ).toNumEqual( expectedPoints[ i ].z );

				} );

			} );

			// curve.type = 'catmullrom'; only from here on
			test( 'getLength/getLengths', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';

				const length = curve.getLength();
				const expectedLength = 551.549686276872;

				expect( length ).toNumEqual( expectedLength );

				const expectedLengths = [
					0,
					120,
					220,
					416.9771560359221,
					536.9771560359221
				];
				const lengths = curve.getLengths( expectedLengths.length - 1 );

				expect( lengths.length ).toBe( expectedLengths.length );

				lengths.forEach( function ( segment, i ) {

					expect( segment ).toNumEqual( expectedLengths[ i ] );

				} );

			} );

			test( 'getPointAt', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';

				const expectedPoints = [
					new Vector3( - 60, - 100, 60 ),
					new Vector3( - 64.84177333183106, 64.86956465359813, 64.84177333183106 ),
					new Vector3( - 28.288507045700854, 104.83101184518996, 28.288507045700854 ),
					new Vector3( 60, - 100, - 60 )
				];

				const points = [
					curve.getPointAt( 0, new Vector3() ),
					curve.getPointAt( 0.3, new Vector3() ),
					curve.getPointAt( 0.5, new Vector3() ),
					curve.getPointAt( 1, new Vector3() )
				];

				expect( points ).toEqual( expectedPoints );

			} );

			test( 'getTangent/getTangentAt', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';

				let expectedTangents = [
					new Vector3( 0, 1, 0 ),
					new Vector3( - 0.0001090274561657922, 0.9999999881130137, 0.0001090274561657922 ),
					new Vector3( 0.7071067811865475, - 2.0930381713877622e-13, - 0.7071067811865475 ),
					new Vector3( 0.43189437062802816, - 0.7917919583070032, - 0.43189437062802816 ),
					new Vector3( - 0.00019991333100812723, - 0.9999999600346592, 0.00019991333100812723 )
				];

				let tangents = [
					curve.getTangent( 0, new Vector3() ),
					curve.getTangent( 0.25, new Vector3() ),
					curve.getTangent( 0.5, new Vector3() ),
					curve.getTangent( 0.75, new Vector3() ),
					curve.getTangent( 1, new Vector3() )
				];

				expectedTangents.forEach( function ( exp, i ) {

					const tangent = tangents[ i ];

					expect( tangent.x ).toNumEqual( exp.x );
					expect( tangent.y ).toNumEqual( exp.y );

				} );

				//

				expectedTangents = [
					new Vector3( 0, 1, 0 ),
					new Vector3( - 0.10709018822205997, 0.9884651653817284, 0.10709018822205997 ),
					new Vector3( 0.6396363672964268, - 0.4262987629159402, - 0.6396363672964268 ),
					new Vector3( 0.5077298411616501, - 0.6960034603275557, - 0.5077298411616501 ),
					new Vector3( - 0.00019991333100812723, - 0.9999999600346592, 0.00019991333100812723 )
				];

				tangents = [
					curve.getTangentAt( 0, new Vector3() ),
					curve.getTangentAt( 0.25, new Vector3() ),
					curve.getTangentAt( 0.5, new Vector3() ),
					curve.getTangentAt( 0.75, new Vector3() ),
					curve.getTangentAt( 1, new Vector3() )
				];

				expectedTangents.forEach( function ( exp, i ) {

					const tangent = tangents[ i ];

					expect( tangent.x ).toNumEqual( exp.x );
					expect( tangent.y ).toNumEqual( exp.y );

				} );

			} );

			test( 'computeFrenetFrames', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';

				const expected = {
					binormals: [
						new Vector3( - 1, 0, 0 ),
						new Vector3( - 0.28685061854203, 0.6396363672964267, - 0.7131493814579701 ),
						new Vector3( - 1.9982670528160395e-8, - 0.0001999133310081272, - 0.9999999800173295 )
					],
					normals: [
						new Vector3( 0, 0, - 1 ),
						new Vector3( - 0.7131493814579699, - 0.6396363672964268, - 0.2868506185420297 ),
						new Vector3( - 0.9999999800173294, 0.00019991333100810582, - 1.99826701852146e-8 )
					],
					tangents: [
						new Vector3( 0, 1, 0 ),
						new Vector3( 0.6396363672964269, - 0.4262987629159403, - 0.6396363672964269 ),
						new Vector3( - 0.0001999133310081273, - 0.9999999600346594, 0.0001999133310081273 )
					]
				};

				const frames = curve.computeFrenetFrames( 2, false );

				Object.keys( expected ).forEach( function ( group ) {

					expected[ group ].forEach( function ( vec, j ) {

						expect( frames[ group ][ j ].x ).toNumEqual( vec.x );
						expect( frames[ group ][ j ].y ).toNumEqual( vec.y );
						expect( frames[ group ][ j ].z ).toNumEqual( vec.z );

					} );

				} );

			} );

			test( 'getUtoTmapping', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';

				const start = curve.getUtoTmapping( 0, 0 );
				const end = curve.getUtoTmapping( 0, curve.getLength() );
				const somewhere = curve.getUtoTmapping( 0.5, 500 );

				const expectedSomewhere = 0.8964116382083199;

				expect( start ).toBe( 0 );
				expect( end ).toBe( 1 );
				expect( somewhere ).toNumEqual( expectedSomewhere );

			} );

			test( 'getSpacedPoints', () => {

				const curve = new CatmullRomCurve3( positions );
				curve.curveType = 'catmullrom';

				const expectedPoints = [
					new Vector3( - 60, - 100, 60 ),
					new Vector3( - 60, 10.311489426555056, 60 ),
					new Vector3( - 65.05889864636504, 117.99691802595966, 65.05889864636504 ),
					new Vector3( 6.054276900088592, 78.7153118386369, - 6.054276900088592 ),
					new Vector3( 64.9991491385602, 8.386980812799566, - 64.9991491385602 ),
					new Vector3( 60, - 100, - 60 )
				];

				const points = curve.getSpacedPoints();

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints );

			} );

			test( 'two points', () => {

				const curve = new CatmullRomCurve3( [
					new Vector3( 0, 0, 0 ),
					new Vector3( 10, 0, 0 )
				], false, 'catmullrom' );

				const expectedPoints = [
					new Vector3( 0, 0, 0 ),
					new Vector3( 5, 0, 0 ),
					new Vector3( 10, 0, 0 )
				];

				const points = curve.getPoints( 2 );

				expect( points.length ).toBe( expectedPoints.length );
				expect( points ).toEqual( expectedPoints );

			} );

		} );

	} );

} );

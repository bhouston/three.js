import { describe, test, expect, beforeEach } from 'vitest';
import { CylinderGeometry } from '@src/geometries/CylinderGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'CylinderGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radiusTop: 10,
				radiusBottom: 20,
				height: 30,
				radialSegments: 20,
				heightSegments: 30,
				openEnded: true,
				thetaStart: 0.1,
				thetaLength: 2.0,
			};

			geometries = [
				new CylinderGeometry(),
				new CylinderGeometry( parameters.radiusTop ),
				new CylinderGeometry( parameters.radiusTop, parameters.radiusBottom ),
				new CylinderGeometry( parameters.radiusTop, parameters.radiusBottom, parameters.height ),
				new CylinderGeometry( parameters.radiusTop, parameters.radiusBottom, parameters.height, parameters.radialSegments ),
				new CylinderGeometry( parameters.radiusTop, parameters.radiusBottom, parameters.height, parameters.radialSegments, parameters.heightSegments ),
				new CylinderGeometry( parameters.radiusTop, parameters.radiusBottom, parameters.height, parameters.radialSegments, parameters.heightSegments, parameters.openEnded ),
				new CylinderGeometry( parameters.radiusTop, parameters.radiusBottom, parameters.height, parameters.radialSegments, parameters.heightSegments, parameters.openEnded, parameters.thetaStart ),
				new CylinderGeometry( parameters.radiusTop, parameters.radiusBottom, parameters.height, parameters.radialSegments, parameters.heightSegments, parameters.openEnded, parameters.thetaStart, parameters.thetaLength ),
			];

		} );

		test( 'Extending', () => {

			const object = new CylinderGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CylinderGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new CylinderGeometry();
			expect( object.type === 'CylinderGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

import { describe, test, expect, beforeEach } from 'vitest';
import { TorusKnotGeometry } from '@src/geometries/TorusKnotGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'TorusKnotGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radius: 10,
				tube: 20,
				tubularSegments: 30,
				radialSegments: 10,
				p: 3,
				q: 2
			};

			geometries = [
				new TorusKnotGeometry(),
				new TorusKnotGeometry( parameters.radius ),
				new TorusKnotGeometry( parameters.radius, parameters.tube ),
				new TorusKnotGeometry( parameters.radius, parameters.tube, parameters.tubularSegments ),
				new TorusKnotGeometry( parameters.radius, parameters.tube, parameters.tubularSegments, parameters.radialSegments ),
				new TorusKnotGeometry( parameters.radius, parameters.tube, parameters.tubularSegments, parameters.radialSegments, parameters.p, parameters.q ),
			];

		} );

		test( 'Extending', () => {

			const object = new TorusKnotGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new TorusKnotGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new TorusKnotGeometry();
			expect( object.type === 'TorusKnotGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

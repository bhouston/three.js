import { describe, test, expect, beforeEach } from 'vitest';
import { CapsuleGeometry } from '@src/geometries/CapsuleGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'CapsuleGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radius: 2,
				height: 2,
				capSegments: 20,
				radialSegments: 20,
				heightSegments: 1
			};

			geometries = [
				new CapsuleGeometry(),
				new CapsuleGeometry( parameters.radius ),
				new CapsuleGeometry( parameters.radius, parameters.height ),
				new CapsuleGeometry( parameters.radius, parameters.height, parameters.capSegments ),
				new CapsuleGeometry( parameters.radius, parameters.height, parameters.capSegments, parameters.radialSegments ),
				new CapsuleGeometry( parameters.radius, parameters.height, parameters.capSegments, parameters.radialSegments, parameters.heightSegments )
			];

		} );

		test( 'Extending', () => {

			const object = new CapsuleGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CapsuleGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new CapsuleGeometry();
			expect( object.type === 'CapsuleGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

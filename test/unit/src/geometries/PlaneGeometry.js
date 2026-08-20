import { describe, test, expect, beforeEach } from 'vitest';
import { PlaneGeometry } from '@src/geometries/PlaneGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'PlaneGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				width: 10,
				height: 30,
				widthSegments: 3,
				heightSegments: 5
			};

			geometries = [
				new PlaneGeometry(),
				new PlaneGeometry( parameters.width ),
				new PlaneGeometry( parameters.width, parameters.height ),
				new PlaneGeometry( parameters.width, parameters.height, parameters.widthSegments ),
				new PlaneGeometry( parameters.width, parameters.height, parameters.widthSegments, parameters.heightSegments ),
			];

		} );

		test( 'Extending', () => {

			const object = new PlaneGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PlaneGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new PlaneGeometry();
			expect( object.type === 'PlaneGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

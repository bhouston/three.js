import { describe, test, expect, beforeEach } from 'vitest';
import { OctahedronGeometry } from '@src/geometries/OctahedronGeometry.js';

import { PolyhedronGeometry } from '@src/geometries/PolyhedronGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'OctahedronGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radius: 10,
				detail: undefined
			};

			geometries = [
				new OctahedronGeometry(),
				new OctahedronGeometry( parameters.radius ),
				new OctahedronGeometry( parameters.radius, parameters.detail ),
			];

		} );

		test( 'Extending', () => {

			const object = new OctahedronGeometry();
			expect( object instanceof PolyhedronGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new OctahedronGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new OctahedronGeometry();
			expect( object.type === 'OctahedronGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

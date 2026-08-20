import { describe, test, expect, beforeEach } from 'vitest';
import { TetrahedronGeometry } from '@src/geometries/TetrahedronGeometry.js';

import { PolyhedronGeometry } from '@src/geometries/PolyhedronGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'TetrahedronGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radius: 10,
				detail: undefined
			};

			geometries = [
				new TetrahedronGeometry(),
				new TetrahedronGeometry( parameters.radius ),
				new TetrahedronGeometry( parameters.radius, parameters.detail ),
			];

		} );

		test( 'Extending', () => {

			const object = new TetrahedronGeometry();
			expect( object instanceof PolyhedronGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new TetrahedronGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new TetrahedronGeometry();
			expect( object.type === 'TetrahedronGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

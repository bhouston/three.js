import { describe, test, expect, beforeEach } from 'vitest';
import { ConeGeometry } from '@src/geometries/ConeGeometry.js';

import { CylinderGeometry } from '@src/geometries/CylinderGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'ConeGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			geometries = [
				new ConeGeometry(),
			];

		} );

		test( 'Extending', () => {

			const object = new ConeGeometry();
			expect( object instanceof CylinderGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ConeGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new ConeGeometry();
			expect( object.type === 'ConeGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

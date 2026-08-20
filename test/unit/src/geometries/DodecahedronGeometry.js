import { describe, test, expect, beforeEach } from 'vitest';
import { DodecahedronGeometry } from '@src/geometries/DodecahedronGeometry.js';

import { PolyhedronGeometry } from '@src/geometries/PolyhedronGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'DodecahedronGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radius: 10,
				detail: undefined
			};

			geometries = [
				new DodecahedronGeometry(),
				new DodecahedronGeometry( parameters.radius ),
				new DodecahedronGeometry( parameters.radius, parameters.detail ),
			];

		} );

		test( 'Extending', () => {

			const object = new DodecahedronGeometry();
			expect( object instanceof PolyhedronGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new DodecahedronGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new DodecahedronGeometry();
			expect( object.type === 'DodecahedronGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

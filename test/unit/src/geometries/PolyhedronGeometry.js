import { describe, test, expect, beforeEach } from 'vitest';
import { PolyhedronGeometry } from '@src/geometries/PolyhedronGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'PolyhedronGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const vertices = [
				1, 1, 1, 	- 1, - 1, 1, 	- 1, 1, - 1, 	1, - 1, - 1
			];

			const indices = [
				2, 1, 0, 	0, 3, 2,	1, 3, 0,	2, 3, 1
			];

			geometries = [
				new PolyhedronGeometry( vertices, indices ),
			];

		} );

		test( 'Extending', () => {

			const object = new PolyhedronGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PolyhedronGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new PolyhedronGeometry();
			expect( object.type === 'PolyhedronGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

import { describe, test, expect, beforeEach } from 'vitest';
import { LatheGeometry } from '@src/geometries/LatheGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'LatheGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				points: [],
				segments: 0,
				phiStart: 0,
				phiLength: 0
			};

			geometries = [
				new LatheGeometry( parameters.points ),
			];

		} );

		test( 'Extending', () => {

			const object = new LatheGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new LatheGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new LatheGeometry();
			expect( object.type === 'LatheGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

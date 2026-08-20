import { describe, test, expect, beforeEach } from 'vitest';
import { CircleGeometry } from '@src/geometries/CircleGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'CircleGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radius: 10,
				segments: 20,
				thetaStart: 0.1,
				thetaLength: 0.2
			};

			geometries = [
				new CircleGeometry(),
				new CircleGeometry( parameters.radius ),
				new CircleGeometry( parameters.radius, parameters.segments ),
				new CircleGeometry( parameters.radius, parameters.segments, parameters.thetaStart ),
				new CircleGeometry( parameters.radius, parameters.segments, parameters.thetaStart, parameters.thetaLength ),
			];

		} );

		test( 'Extending', () => {

			const object = new CircleGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new CircleGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new CircleGeometry();
			expect( object.type === 'CircleGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

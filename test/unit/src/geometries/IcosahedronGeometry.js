import { describe, test, expect, beforeEach } from 'vitest';
import { IcosahedronGeometry } from '@src/geometries/IcosahedronGeometry.js';

import { PolyhedronGeometry } from '@src/geometries/PolyhedronGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'IcosahedronGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				radius: 10,
				detail: undefined
			};

			geometries = [
				new IcosahedronGeometry(),
				new IcosahedronGeometry( parameters.radius ),
				new IcosahedronGeometry( parameters.radius, parameters.detail ),
			];

		} );

		test( 'Extending', () => {

			const object = new IcosahedronGeometry();
			expect( object instanceof PolyhedronGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new IcosahedronGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new IcosahedronGeometry();
			expect( object.type === 'IcosahedronGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

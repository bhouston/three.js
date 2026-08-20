import { describe, test, expect, beforeEach } from 'vitest';

import { BoxHelper } from '@src/helpers/BoxHelper.js';

import { LineSegments } from '@src/objects/LineSegments.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';
import { BoxGeometry } from '@src/geometries/BoxGeometry.js';
import { SphereGeometry } from '@src/geometries/SphereGeometry.js';
import { Mesh } from '@src/objects/Mesh.js';

describe( 'Helpers', () => {

	describe( 'BoxHelper', () => {

		let geometries = undefined;

		beforeEach( () => {

			// Test with a normal cube and a box helper
			const boxGeometry = new BoxGeometry();
			const box = new Mesh( boxGeometry );
			const boxHelper = new BoxHelper( box );

			// The same should happen with a comparable sphere
			const sphereGeometry = new SphereGeometry();
			const sphere = new Mesh( sphereGeometry );
			const sphereBoxHelper = new BoxHelper( sphere );

			// Note that unlike what I'd like to, these doesn't check the equivalency
			// of the two generated geometries
			geometries = [ boxHelper.geometry, sphereBoxHelper.geometry ];

		} );

		test( 'Extending', () => {

			const object = new BoxHelper();
			expect( object instanceof LineSegments ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new BoxHelper();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new BoxHelper();
			expect( object.type === 'BoxHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new BoxHelper();
			object.dispose();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

	} );

} );

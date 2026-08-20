import { describe, test, expect, beforeEach } from 'vitest';
import { BoxGeometry } from '@src/geometries/BoxGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { runStdGeometryTests } from '@test-utils/std-geometry-tests.js';

describe( 'Geometries', () => {

	describe( 'BoxGeometry', () => {

		let geometries = undefined;
		beforeEach( () => {

			const parameters = {
				width: 10,
				height: 20,
				depth: 30,
				widthSegments: 2,
				heightSegments: 3,
				depthSegments: 4
			};

			geometries = [
				new BoxGeometry(),
				new BoxGeometry( parameters.width, parameters.height, parameters.depth ),
				new BoxGeometry( parameters.width, parameters.height, parameters.depth, parameters.widthSegments, parameters.heightSegments, parameters.depthSegments ),
			];

		} );

		test( 'Extending', () => {

			const object = new BoxGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new BoxGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new BoxGeometry();
			expect( object.type === 'BoxGeometry' ).toBeTruthy();

		} );

		test( 'Standard geometry tests', () => {

			runStdGeometryTests( geometries );

		} );

		test( 'toJSON: parametric serialization for untransformed geometry', () => {

			const geometry = new BoxGeometry( 10, 20, 30 );
			const json = geometry.toJSON();

			expect( geometry._transformed ).toBe( false );
			expect( json.type ).toBe( 'BoxGeometry' );
			expect( json.width ).toBe( 10 );
			expect( json.height ).toBe( 20 );
			expect( json.depth ).toBe( 30 );
			expect( json.data ).toBe( undefined );

		} );

		test( 'toJSON: attribute serialization after translate()', () => {

			const geometry = new BoxGeometry( 10, 20, 30 );
			geometry.translate( 1, 2, 3 );
			const json = geometry.toJSON();

			expect( geometry._transformed ).toBe( true );
			expect( json.type ).toBe( 'BufferGeometry' );
			expect( json.width ).toBe( undefined );
			expect( json.data && json.data.attributes && json.data.attributes.position ).toBeTruthy();

		} );

		test( 'toJSON: clone of a transformed geometry preserves _transformed', () => {

			const source = new BoxGeometry( 10, 20, 30 );
			source.translate( 1, 2, 3 );
			const clone = source.clone();
			const json = clone.toJSON();

			expect( clone._transformed ).toBe( true );
			expect( json.type ).toBe( 'BufferGeometry' );
			expect( json.data && json.data.attributes && json.data.attributes.position ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect, beforeEach } from 'vitest';
import { ShapeGeometry } from '@src/geometries/ShapeGeometry.js';

import { Shape } from '@src/extras/core/Shape.js';
import { BufferGeometry } from '@src/core/BufferGeometry.js';

describe( 'Geometries', () => {

	describe( 'ShapeGeometry', () => {

		let geometries = undefined; // eslint-disable-line no-unused-vars
		beforeEach( () => {

			const triangleShape = new Shape();
			triangleShape.moveTo( 0, - 1 );
			triangleShape.lineTo( 1, 1 );
			triangleShape.lineTo( - 1, 1 );

			geometries = [
				new ShapeGeometry( triangleShape ),
			];

		} );

		test( 'Extending', () => {

			const object = new ShapeGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ShapeGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new ShapeGeometry();
			expect( object.type === 'ShapeGeometry' ).toBeTruthy();

		} );

	} );

} );

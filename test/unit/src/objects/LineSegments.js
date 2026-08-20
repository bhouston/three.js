import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';
import { Line } from '@src/objects/Line.js';
import { LineSegments } from '@src/objects/LineSegments.js';

describe( 'Objects', () => {

	describe( 'LineSegments', () => {

		test( 'Extending', () => {

			const lineSegments = new LineSegments();
			expect( lineSegments instanceof Object3D ).toBe( true );
			expect( lineSegments instanceof Line ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new LineSegments();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new LineSegments();
			expect( object.type === 'LineSegments' ).toBeTruthy();

		} );

		test( 'isLineSegments', () => {

			const object = new LineSegments();
			expect( object.isLineSegments ).toBeTruthy();

		} );

	} );

} );

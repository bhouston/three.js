import { describe, test, expect } from 'vitest';
import { ExtrudeGeometry } from '@src/geometries/ExtrudeGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';

describe( 'Geometries', () => {

	describe( 'ExtrudeGeometry', () => {

		test( 'Extending', () => {

			const object = new ExtrudeGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new ExtrudeGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new ExtrudeGeometry();
			expect( object.type === 'ExtrudeGeometry' ).toBeTruthy();

		} );

	} );

} );

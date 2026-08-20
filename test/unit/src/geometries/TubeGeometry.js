import { describe, test, expect, beforeEach } from 'vitest';
import { TubeGeometry } from '@src/geometries/TubeGeometry.js';

import { LineCurve3 } from '@src/extras/curves/LineCurve3.js';
import { Vector3 } from '@src/math/Vector3.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';

describe( 'Geometries', () => {

	describe( 'TubeGeometry', () => {

		let geometries = undefined; // eslint-disable-line no-unused-vars
		beforeEach( () => {

			const path = new LineCurve3( new Vector3( 0, 0, 0 ), new Vector3( 0, 1, 0 ) );

			geometries = [
				new TubeGeometry( path ),
			];

		} );

		test( 'Extending', () => {

			const object = new TubeGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new TubeGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new TubeGeometry();
			expect( object.type === 'TubeGeometry' ).toBeTruthy();

		} );

	} );

} );

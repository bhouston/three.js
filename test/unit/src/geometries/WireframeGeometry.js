import { describe, test, expect, beforeEach } from 'vitest';
import { WireframeGeometry } from '@src/geometries/WireframeGeometry.js';

import { BufferGeometry } from '@src/core/BufferGeometry.js';

describe( 'Geometries', () => {

	describe( 'WireframeGeometry', () => {

		let geometries = undefined; // eslint-disable-line no-unused-vars
		beforeEach( () => {

			geometries = [
				new WireframeGeometry()
			];

		} );

		test( 'Extending', () => {

			const object = new WireframeGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new WireframeGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new WireframeGeometry();
			expect( object.type === 'WireframeGeometry' ).toBeTruthy();

		} );

	} );

} );

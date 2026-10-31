import { describe, expect, test } from 'vitest';
import { assert } from '../../utils/assertions.js';

import { AxesHelper } from '../../../../src/helpers/AxesHelper.js';

import { LineSegments } from '../../../../src/objects/LineSegments.js';

describe( 'Helpers', () => {

	describe( 'AxesHelper', () => {

		// INHERITANCE
		test( 'Extending', () => {

			const object = new AxesHelper();
			assert.strictEqual(
				object instanceof LineSegments, true,
				'AxesHelper extends from LineSegments'
			);

		} );

		// INSTANCING
		test( 'Instancing', () => {

			const object = new AxesHelper();
			assert.ok( object, 'Can instantiate an AxesHelper.' );

		} );

		// PROPERTIES
		test( 'type', () => {

			const object = new AxesHelper();
			assert.ok(
				object.type === 'AxesHelper',
				'AxesHelper.type should be AxesHelper'
			);

		} );

		// PUBLIC

		test( 'dispose', () => {

			expect.assertions( 0 );

			const object = new AxesHelper();
			object.dispose();

		} );

	} );

} );

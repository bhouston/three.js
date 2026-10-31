import { describe, test } from 'vitest';
import { assert } from '../../utils/assertions.js';

import { LoaderUtils } from '../../../../src/loaders/LoaderUtils.js';

describe( 'Loaders', () => {

	describe( 'LoaderUtils', () => {

		// STATIC

		test( 'extractUrlBase', () => {

			assert.equal( '/path/to/', LoaderUtils.extractUrlBase( '/path/to/model.glb' ) );
			assert.equal( './', LoaderUtils.extractUrlBase( 'model.glb' ) );
			assert.equal( '/', LoaderUtils.extractUrlBase( '/model.glb' ) );

		} );

	} );

} );

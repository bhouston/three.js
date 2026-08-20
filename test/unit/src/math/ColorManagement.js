import { describe, test, expect } from 'vitest';
import { ColorManagement } from '@src/math/ColorManagement.js';

describe( 'Maths', () => {

	describe( 'ColorManagement', () => {

		test( 'enabled', () => {

			expect( ColorManagement.enabled ).toBe( true );

		} );

	} );

} );

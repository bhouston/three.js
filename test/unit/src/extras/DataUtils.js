import { describe, test, expect } from 'vitest';
import * as DataUtils from '@src/extras/DataUtils.js';

import { CONSOLE_LEVEL } from '@test-utils/console-wrapper.js';

describe( 'Extras', () => {

	describe( 'DataUtils', () => {

		// PUBLIC
		test( 'toHalfFloat', () => {

			expect( DataUtils.toHalfFloat( 0 ) === 0 ).toBeTruthy();

			// suppress the following console message during testing
			// THREE.DataUtils.toHalfFloat(): Value out of range.

			console.level = CONSOLE_LEVEL.OFF;
			expect( DataUtils.toHalfFloat( 100000 ) === 31743 ).toBeTruthy();
			expect( DataUtils.toHalfFloat( - 100000 ) === 64511 ).toBeTruthy();
			console.level = CONSOLE_LEVEL.DEFAULT;

			expect( DataUtils.toHalfFloat( 65504 ) === 31743 ).toBeTruthy();
			expect( DataUtils.toHalfFloat( - 65504 ) === 64511 ).toBeTruthy();
			expect( DataUtils.toHalfFloat( Math.PI ) === 16968 ).toBeTruthy();
			expect( DataUtils.toHalfFloat( - Math.PI ) === 49736 ).toBeTruthy();

		} );

		test( 'fromHalfFloat', () => {

			expect( DataUtils.fromHalfFloat( 0 ) === 0 ).toBeTruthy();
			expect( DataUtils.fromHalfFloat( 31744 ) === Infinity ).toBeTruthy();
			expect( DataUtils.fromHalfFloat( 64512 ) === - Infinity ).toBeTruthy();
			expect( DataUtils.fromHalfFloat( 31743 ) === 65504 ).toBeTruthy();
			expect( DataUtils.fromHalfFloat( 64511 ) === - 65504 ).toBeTruthy();
			expect( DataUtils.fromHalfFloat( 16968 ) === 3.140625 ).toBeTruthy();
			expect( DataUtils.fromHalfFloat( 49736 ) === - 3.140625 ).toBeTruthy();

		} );

	} );

} );

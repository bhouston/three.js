// Vitest setup: matchers with the same names/semantics as the custom QUnit
// assertions in test/unit/utils/qunit-utils.js, so a segment conversion is a
// mechanical find/replace rather than a rewrite of the assertion logic:
//
//   assert.numEqual( a, b )        -> expect( a ).toNumEqual( b )
//   assert.equalKey( obj, ref, k ) -> expect( obj ).toEqualKey( ref, k )
//   assert.smartEqual( a, b )      -> expect( a ).toSmartEqual( b )
//
// Everything else (assert.ok, assert.strictEqual, assert.deepEqual, ...) maps
// directly onto vitest's built-in expect() API and needs no shim.

import { expect } from 'vitest';
import { SmartComparer } from './utils/SmartComparer.js';

expect.extend( {

	toNumEqual( actual, expected ) {

		const diff = Math.abs( actual - expected );
		const pass = diff < 0.1;

		return {
			pass,
			message: () => `${ actual } should${ pass ? ' not' : '' } be equal to ${ expected }`,
		};

	},

	toEqualKey( obj, ref, key ) {

		const actual = obj[ key ];
		const expected = ref[ key ];
		const pass = actual == expected;

		return {
			pass,
			message: () => `${ actual } should${ pass ? ' not' : '' } be equal to ${ expected } for key "${ key }"`,
		};

	},

	toSmartEqual( actual, expected ) {

		const cmp = new SmartComparer();
		const pass = cmp.areEqual( actual, expected );

		return {
			pass,
			message: () => cmp.getDiagnostic() || `expected values to${ pass ? ' not' : '' } be smart-equal`,
		};

	},

} );

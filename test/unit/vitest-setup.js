// Vitest setup: matchers with the same names/semantics as the custom QUnit
// assertions in test/unit/utils/qunit-utils.js, so a segment conversion is a
// mechanical find/replace rather than a rewrite of the assertion logic:
//
//   assert.numEqual( a, b )        -> expect( a ).toNumEqual( b )
//   assert.equalKey( obj, ref, k ) -> expect( obj ).toEqualKey( ref, k )
//   assert.smartEqual( a, b )      -> expect( a ).toSmartEqual( b )
//
// Everything else (assert.ok, assert.strictEqual, ...) maps directly onto
// vitest's built-in expect() API - except assert.deepEqual on whole three.js
// object graphs (e.g. comparing two Object3D instances), which needs
// toEqualLikeQUnit below instead of plain toEqual. Two behaviors QUnit's
// deepEqual has that vitest's toEqual doesn't:
//   - numbers compare with `===`, so -0 and 0 are equal (vitest's toEqual
//     uses Object.is, which correctly-for-NaN's-sake treats them as
//     different - but three.js code legitimately produces -0 in places,
//     e.g. rotation round-trips through quaternions).
//   - two functions are always treated as equal, so per-instance bound
//     callbacks (e.g. Euler's onChange, bound to a specific Object3D in its
//     constructor) don't make two otherwise-identical instances look
//     unequal just because their callback closures differ.

import { expect } from 'vitest';
import { SmartComparer } from './utils/SmartComparer.js';

function normalizeForQUnitStyleEqual( value, seen = new WeakMap() ) {

	if ( typeof value === 'number' ) return value === 0 ? 0 : value;
	if ( typeof value === 'function' ) return undefined;
	if ( value === null || typeof value !== 'object' ) return value;
	if ( seen.has( value ) ) return seen.get( value );

	const clone = Array.isArray( value ) ? [] : Object.create( Object.getPrototypeOf( value ) );
	seen.set( value, clone );

	for ( const key of Object.keys( value ) ) {

		clone[ key ] = normalizeForQUnitStyleEqual( value[ key ], seen );

	}

	return clone;

}

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

	toEqualLikeQUnit( actual, expected ) {

		const pass = this.equals(
			normalizeForQUnitStyleEqual( actual ),
			normalizeForQUnitStyleEqual( expected ),
		);

		return {
			pass,
			message: () => `expected values to${ pass ? ' not' : '' } be structurally equal (QUnit deepEqual semantics: -0 equals 0, functions always equal)`,
		};

	},

} );

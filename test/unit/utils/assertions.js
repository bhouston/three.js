import { assert as vitestAssert, expect } from 'vitest';
import { SmartComparer } from './SmartComparer.js';

// Object copies retain their enumerable data and instance methods. Signed zero
// and separately created functions with identical source represent the same data.
expect.extend( {

	toEqualProperties( actual, expected ) {

		const pass = this.equals( actual, expected, [ ( a, b ) => {

			if ( a === 0 && b === 0 ) return true;
			if ( typeof a === 'function' && typeof b === 'function' ) return a.toString() === b.toString();
			if ( a !== null && b !== null && typeof a === 'object' && typeof b === 'object' && a.constructor !== b.constructor ) return false;

		} ], true );

		return { pass, actual, expected, message: () => 'Expected objects to have equal enumerable properties' };

	}

} );

// Keep the domain comparisons used by the geometry and math tests together.
const assert = {

	...vitestAssert,

	equalProperties( actual, expected, message ) {

		expect( actual, message ).toEqualProperties( expected );

	},

	numEqual( actual, expected, message ) {

		vitestAssert.isBelow( Math.abs( actual - expected ), 0.1, message );

	},

	equalKey( actual, expected, key ) {

		vitestAssert.equal( actual[ key ], expected[ key ], `Property "${ key }"` );

	},

	smartEqual( actual, expected, message ) {

		const comparer = new SmartComparer();
		const equal = comparer.areEqual( actual, expected );
		vitestAssert.isTrue( equal, comparer.getDiagnostic() || message );

	},

	propEqual( actual, expected, message ) {

		vitestAssert.deepEqual( { ...actual }, { ...expected }, message );

	},

	notPropEqual( actual, expected, message ) {

		vitestAssert.notDeepEqual( { ...actual }, { ...expected }, message );

	}

};

export { assert };

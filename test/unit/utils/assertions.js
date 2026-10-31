import { assert as vitestAssert } from 'vitest';
import { SmartComparer } from './SmartComparer.js';

// Keep the domain comparisons used by the geometry and math tests together.
const assert = {

	...vitestAssert,

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

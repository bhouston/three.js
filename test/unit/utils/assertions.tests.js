import { describe, expect, test } from 'vitest';
import { Vector3 } from '../../../src/math/Vector3.js';
import { assert } from './assertions.js';

describe( 'Property comparisons', () => {

	test( 'compares nested identifiers and additional class properties', () => {

		const actual = { userData: { id: 1, uuid: 'first' }, vector: new Vector3() };
		const expected = { userData: { id: 1, uuid: 'first' }, vector: new Vector3() };
		assert.equalProperties( actual, expected );

		actual.userData.id = 2;
		expect( () => assert.equalProperties( actual, expected ) ).toThrow();
		actual.userData.id = 1;
		actual.userData.uuid = 'second';
		expect( () => assert.equalProperties( actual, expected ) ).toThrow();
		actual.userData.uuid = 'first';
		actual.vector.additional = 1;
		expect( () => assert.equalProperties( actual, expected ) ).toThrow();

	} );

	test( 'normalizes signed zero and separately created instance methods', () => {

		const create = value => ( { value, method() {

			return this.value;

		} } );
		assert.equalProperties( create( - 0 ), create( 0 ) );
		expect( () => assert.equalProperties( create( 0 ), { value: 0, method() {

			return 1;

		} } ) ).toThrow();

	} );

	test( 'handles cycles without ignoring mismatches or missing properties', () => {

		const actual = { value: 1, array: [] };
		const expected = { value: 1, array: [] };
		actual.array.push( actual.array, actual );
		expected.array.push( expected.array, expected );
		assert.equalProperties( actual, expected );
		actual.value = 2;
		expect( () => assert.equalProperties( actual, expected ) ).toThrow();
		actual.value = 1;
		actual.additional = undefined;
		expect( () => assert.equalProperties( actual, expected ) ).toThrow();

	} );

} );

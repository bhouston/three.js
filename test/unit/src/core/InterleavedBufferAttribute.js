import { describe, test, expect } from 'vitest';

import { InterleavedBufferAttribute } from '@src/core/InterleavedBufferAttribute.js';
import { InterleavedBuffer } from '@src/core/InterleavedBuffer.js';

describe( 'Core', () => {

	describe( 'InterleavedBufferAttribute', () => {

		test( 'Instancing', () => {

			const object = new InterleavedBufferAttribute();
			expect( object ).toBeTruthy();

		} );

		test( 'count', () => {

			const buffer = new InterleavedBuffer( new Float32Array( [ 1, 2, 3, 7, 8, 9 ] ), 3 );
			const instance = new InterleavedBufferAttribute( buffer, 2, 0 );

			expect( instance.count === 2 ).toBeTruthy();

		} );

		test( 'isInterleavedBufferAttribute', () => {

			const object = new InterleavedBufferAttribute();
			expect( object.isInterleavedBufferAttribute ).toBeTruthy();

		} );

		// setY, setZ and setW are calculated in the same way so not testing this
		// TODO: ( you can't be sure that will be the case in future, or a mistake was introduce in one off them ! )
		test( 'setX', () => {

			let buffer = new InterleavedBuffer( new Float32Array( [ 1, 2, 3, 7, 8, 9 ] ), 3 );
			let instance = new InterleavedBufferAttribute( buffer, 2, 0 );

			instance.setX( 0, 123 );
			instance.setX( 1, 321 );

			expect( instance.data.array[ 0 ] === 123 &&
				instance.data.array[ 3 ] === 321 ).toBeTruthy();

			buffer = new InterleavedBuffer( new Float32Array( [ 1, 2, 3, 7, 8, 9 ] ), 3 );
			instance = new InterleavedBufferAttribute( buffer, 2, 1 );

			instance.setX( 0, 123 );
			instance.setX( 1, 321 );

			// the offset was defined as 1, so go one step further in the array
			expect( instance.data.array[ 1 ] === 123 &&
				instance.data.array[ 4 ] === 321 ).toBeTruthy();

		} );

	} );

} );

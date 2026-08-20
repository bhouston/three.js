import { describe, test, expect } from 'vitest';

import { InstancedBufferAttribute } from '@src/core/InstancedBufferAttribute.js';
import { BufferAttribute } from '@src/core/BufferAttribute.js';

describe( 'Core', () => {

	describe( 'InstancedBufferAttribute', () => {

		test( 'Extending', () => {

			const object = new BufferAttribute();
			expect( object instanceof BufferAttribute ).toBe( true );

		} );

		test( 'Instancing', () => {

			// array, itemSize
			let instance = new InstancedBufferAttribute( new Float32Array( 10 ), 2 );
			expect( instance.meshPerAttribute === 1 ).toBeTruthy();

			// array, itemSize, normalized, meshPerAttribute
			instance = new InstancedBufferAttribute( new Float32Array( 10 ), 2, false, 123 );
			expect( instance.meshPerAttribute === 123 ).toBeTruthy();

		} );

		test( 'isInstancedBufferAttribute', () => {

			const object = new InstancedBufferAttribute();
			expect( object.isInstancedBufferAttribute ).toBeTruthy();

		} );

		test( 'copy', () => {

			const array = new Float32Array( [ 1, 2, 3, 7, 8, 9 ] );
			const instance = new InstancedBufferAttribute( array, 2, true, 123 );
			const copiedInstance = instance.copy( instance );

			expect( copiedInstance instanceof InstancedBufferAttribute ).toBeTruthy();
			expect( copiedInstance.itemSize === 2 ).toBeTruthy();
			expect( copiedInstance.normalized === true ).toBeTruthy();
			expect( copiedInstance.meshPerAttribute === 123 ).toBeTruthy();

			for ( let i = 0; i < array.length; i ++ ) {

				expect( copiedInstance.array[ i ] === array[ i ] ).toBeTruthy();

			}

		} );

	} );

} );

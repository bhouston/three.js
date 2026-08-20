import { describe, test, expect } from 'vitest';

import { InstancedInterleavedBuffer } from '@src/core/InstancedInterleavedBuffer.js';
import { InterleavedBuffer } from '@src/core/InterleavedBuffer.js';

describe( 'Core', () => {

	describe( 'InstancedInterleavedBuffer', () => {

		test( 'Extending', () => {

			const object = new InstancedInterleavedBuffer();
			expect( object instanceof InterleavedBuffer ).toBe( true );

		} );

		test( 'Instancing', () => {

			const array = new Float32Array( [ 1, 2, 3, 7, 8, 9 ] );
			const instance = new InstancedInterleavedBuffer( array, 3 );

			expect( instance.meshPerAttribute === 1 ).toBeTruthy();

		} );

		test( 'isInstancedInterleavedBuffer', () => {

			const object = new InstancedInterleavedBuffer();
			expect( object.isInstancedInterleavedBuffer ).toBeTruthy();

		} );

		test( 'copy', () => {

			const array = new Float32Array( [ 1, 2, 3, 7, 8, 9 ] );
			const instance = new InstancedInterleavedBuffer( array, 3 );
			const copiedInstance = instance.copy( instance );

			expect( copiedInstance.meshPerAttribute === 1 ).toBeTruthy();

		} );

	} );

} );

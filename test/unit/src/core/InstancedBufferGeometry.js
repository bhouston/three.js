import { describe, test, expect } from 'vitest';

import { InstancedBufferGeometry } from '@src/core/InstancedBufferGeometry.js';
import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { BufferAttribute } from '@src/core/BufferAttribute.js';

describe( 'Core', () => {

	describe( 'InstancedBufferGeometry', () => {

		function createClonableMock() {

			return {
				callCount: 0,
				clone: function () {

					this.callCount ++;
					return this;

				}
			};

		}

		test( 'Extending', () => {

			const object = new InstancedBufferGeometry();
			expect( object instanceof BufferGeometry ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new InstancedBufferGeometry();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new InstancedBufferGeometry();
			expect( object.type === 'InstancedBufferGeometry' ).toBeTruthy();

		} );

		test( 'isInstancedBufferGeometry', () => {

			const object = new InstancedBufferGeometry();
			expect( object.isInstancedBufferGeometry ).toBeTruthy();

		} );

		test( 'copy', () => {

			const instanceMock1 = {};
			const instanceMock2 = {};
			const indexMock = createClonableMock();
			const defaultAttribute1 = new BufferAttribute( new Float32Array( [ 1 ] ) );
			const defaultAttribute2 = new BufferAttribute( new Float32Array( [ 2 ] ) );

			const instance = new InstancedBufferGeometry();

			instance.addGroup( 0, 10, instanceMock1 );
			instance.addGroup( 10, 5, instanceMock2 );
			instance.setIndex( indexMock );
			instance.setAttribute( 'defaultAttribute1', defaultAttribute1 );
			instance.setAttribute( 'defaultAttribute2', defaultAttribute2 );

			const copiedInstance = new InstancedBufferGeometry().copy( instance );

			expect( copiedInstance instanceof InstancedBufferGeometry ).toBeTruthy();

			expect( copiedInstance.index ).toBe( indexMock );
			expect( copiedInstance.index.callCount ).toBe( 1 );

			expect( copiedInstance.attributes[ 'defaultAttribute1' ] instanceof BufferAttribute ).toBeTruthy();
			expect( copiedInstance.attributes[ 'defaultAttribute1' ].array ).toEqual( defaultAttribute1.array );
			expect( copiedInstance.attributes[ 'defaultAttribute2' ].array ).toEqual( defaultAttribute2.array );

			expect( copiedInstance.groups[ 0 ].start ).toBe( 0 );
			expect( copiedInstance.groups[ 0 ].count ).toBe( 10 );
			expect( copiedInstance.groups[ 0 ].materialIndex ).toBe( instanceMock1 );

			expect( copiedInstance.groups[ 1 ].start ).toBe( 10 );
			expect( copiedInstance.groups[ 1 ].count ).toBe( 5 );
			expect( copiedInstance.groups[ 1 ].materialIndex ).toBe( instanceMock2 );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { InstancedMesh } from '@src/objects/InstancedMesh.js';
import { Mesh } from '@src/objects/Mesh.js';

describe( 'Objects', () => {

	describe( 'InstancedMesh', () => {

		test( 'Extending', () => {

			const object = new InstancedMesh();
			expect( object instanceof Mesh ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new InstancedMesh();
			expect( object ).toBeTruthy();

		} );

		test( 'isInstancedMesh', () => {

			const object = new InstancedMesh();
			expect( object.isInstancedMesh ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new InstancedMesh();
			object.dispose();

		} );

	} );

} );

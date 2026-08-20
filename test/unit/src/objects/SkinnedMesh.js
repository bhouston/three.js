import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';
import { Mesh } from '@src/objects/Mesh.js';
import { SkinnedMesh } from '@src/objects/SkinnedMesh.js';
import { AttachedBindMode } from '@src/constants.js';

describe( 'Objects', () => {

	describe( 'SkinnedMesh', () => {

		test( 'Extending', () => {

			const skinnedMesh = new SkinnedMesh();

			expect( skinnedMesh instanceof Object3D ).toBe( true );
			expect( skinnedMesh instanceof Mesh ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new SkinnedMesh();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new SkinnedMesh();
			expect( object.type === 'SkinnedMesh' ).toBeTruthy();

		} );

		test( 'bindMode', () => {

			const object = new SkinnedMesh();
			expect( object.bindMode === AttachedBindMode ).toBeTruthy();

		} );

		test( 'isSkinnedMesh', () => {

			const object = new SkinnedMesh();
			expect( object.isSkinnedMesh ).toBeTruthy();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { Line } from '@src/objects/Line.js';
import { Object3D } from '@src/core/Object3D.js';
import { Material } from '@src/materials/Material.js';

describe( 'Objects', () => {

	describe( 'Line', () => {

		test( 'Extending', () => {

			const line = new Line();
			expect( line instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Line();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Line();
			expect( object.type === 'Line' ).toBeTruthy();

		} );

		test( 'isLine', () => {

			const object = new Line();
			expect( object.isLine ).toBeTruthy();

		} );

		test( 'copy/material', () => {

			// Material arrays are cloned
			const mesh1 = new Line();
			mesh1.material = [ new Material() ];

			const copy1 = mesh1.clone();
			expect( mesh1.material ).not.toBe( copy1.material );

			// Non arrays are not cloned
			const mesh2 = new Line();
			mesh1.material = new Material();
			const copy2 = mesh2.clone();
			expect( mesh2.material ).toBe( copy2.material );

		} );

	} );

} );

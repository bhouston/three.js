import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';
import { Material } from '@src/materials/Material.js';
import { Points } from '@src/objects/Points.js';

describe( 'Objects', () => {

	describe( 'Points', () => {

		test( 'Extending', () => {

			const points = new Points();
			expect( points instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Points();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Points();
			expect( object.type === 'Points' ).toBeTruthy();

		} );

		test( 'isPoints', () => {

			const object = new Points();
			expect( object.isPoints ).toBeTruthy();

		} );

		test( 'copy/material', () => {

			// Material arrays are cloned
			const mesh1 = new Points();
			mesh1.material = [ new Material() ];

			const copy1 = mesh1.clone();
			expect( mesh1.material ).not.toBe( copy1.material );

			// Non arrays are not cloned
			const mesh2 = new Points();
			mesh1.material = new Material();
			const copy2 = mesh2.clone();
			expect( mesh2.material ).toBe( copy2.material );

		} );

	} );

} );

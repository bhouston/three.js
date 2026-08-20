import { describe, test, expect } from 'vitest';
import { Skeleton } from '@src/objects/Skeleton.js';

describe( 'Objects', () => {

	describe( 'Skeleton', () => {

		test( 'Instancing', () => {

			const object = new Skeleton();
			expect( object ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new Skeleton();
			object.dispose();

		} );

	} );

} );

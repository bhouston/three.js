import { describe, test, expect } from 'vitest';

import { SkeletonHelper } from '@src/helpers/SkeletonHelper.js';

import { LineSegments } from '@src/objects/LineSegments.js';
import { Bone } from '@src/objects/Bone.js';

describe( 'Helpers', () => {

	describe( 'SkeletonHelper', () => {

		test( 'Extending', () => {

			const bone = new Bone();
			const object = new SkeletonHelper( bone );
			expect( object instanceof LineSegments ).toBe( true );

		} );

		test( 'Instancing', () => {

			const bone = new Bone();
			const object = new SkeletonHelper( bone );
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const bone = new Bone();
			const object = new SkeletonHelper( bone );
			expect( object.type === 'SkeletonHelper' ).toBeTruthy();

		} );

		test( 'isSkeletonHelper', () => {

			const bone = new Bone();
			const object = new SkeletonHelper( bone );
			expect( object.isSkeletonHelper ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const bone = new Bone();
			const object = new SkeletonHelper( bone );
			object.dispose();

		} );

	} );

} );

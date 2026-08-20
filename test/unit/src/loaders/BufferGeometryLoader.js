import { describe, test, expect } from 'vitest';

import { BufferGeometryLoader } from '@src/loaders/BufferGeometryLoader.js';
import { BufferAttribute } from '@src/core/BufferAttribute.js';
import { BufferGeometry } from '@src/core/BufferGeometry.js';
import { DynamicDrawUsage } from '@src/constants.js';
import { Loader } from '@src/loaders/Loader.js';

describe( 'Loaders', () => {

	describe( 'BufferGeometryLoader', () => {

		test( 'Extending', () => {

			const object = new BufferGeometryLoader();
			expect( object instanceof Loader ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new BufferGeometryLoader();
			expect( object ).toBeTruthy();

		} );

		test( 'parser - attributes - circlable', () => {

			const loader = new BufferGeometryLoader();
			const geometry = new BufferGeometry();
			const attr = new BufferAttribute( new Float32Array( [ 7, 8, 9, 10, 11, 12 ] ), 2, true );
			attr.name = 'attribute';
			attr.setUsage( DynamicDrawUsage );

			geometry.setAttribute( 'attr', attr );

			const geometry2 = loader.parse( geometry.toJSON() );

			expect( geometry2.getAttribute( 'attr' ) ).toBeTruthy();

			expect( geometry.getAttribute( 'attr' ) ).toEqual( geometry2.getAttribute( 'attr' ) );

		} );

	} );

} );

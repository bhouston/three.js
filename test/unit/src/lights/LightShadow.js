import { describe, test, expect } from 'vitest';
import { LightShadow } from '@src/lights/LightShadow.js';
import { OrthographicCamera } from '@src/cameras/OrthographicCamera.js';

describe( 'Lights', () => {

	describe( 'LightShadow', () => {

		test( 'Instancing', () => {

			const camera = new OrthographicCamera( - 5, 5, 5, - 5, 0.5, 500 );
			const object = new LightShadow( camera );
			expect( object ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new LightShadow();
			object.dispose();

		} );

		test( 'clone/copy', () => {

			const a = new LightShadow( new OrthographicCamera( - 5, 5, 5, - 5, 0.5, 500 ) );
			const b = new LightShadow( new OrthographicCamera( - 3, 3, 3, - 3, 0.3, 300 ) );

			expect( a ).not.toEqual( b );

			const c = a.clone();
			expect( a ).toSmartEqual( c );

			c.mapSize.set( 256, 256 );
			expect( a ).not.toEqual( c );

			b.copy( a );
			expect( a ).toSmartEqual( b );

			b.mapSize.set( 512, 512 );
			expect( a ).not.toEqual( b );

		} );

	} );

} );

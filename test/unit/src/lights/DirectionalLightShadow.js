import { describe, test, expect } from 'vitest';
import { DirectionalLightShadow } from '@src/lights/DirectionalLightShadow.js';
import { LightShadow } from '@src/lights/LightShadow.js';
import { ObjectLoader } from '@src/loaders/ObjectLoader.js';
import { DirectionalLight } from '@src/lights/DirectionalLight.js';

describe( 'Lights', () => {

	describe( 'DirectionalLightShadow', () => {

		test( 'Extending', () => {

			const object = new DirectionalLightShadow();
			expect( object instanceof LightShadow ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new DirectionalLightShadow();
			expect( object ).toBeTruthy();

		} );

		test( 'isDirectionalLightShadow', () => {

			const object = new DirectionalLightShadow();
			expect( object.isDirectionalLightShadow ).toBeTruthy();

		} );

		test( 'clone/copy', () => {

			const a = new DirectionalLightShadow();
			const b = new DirectionalLightShadow();

			expect( a ).not.toEqual( b );

			const c = a.clone();
			expect( a ).toSmartEqual( c );

			c.mapSize.set( 1024, 1024 );
			expect( a ).not.toEqual( c );

			b.copy( a );
			expect( a ).toSmartEqual( b );

			b.mapSize.set( 512, 512 );
			expect( a ).not.toEqual( b );

		} );

		test( 'toJSON', () => {

			const light = new DirectionalLight();
			const shadow = new DirectionalLightShadow();

			shadow.bias = 10;
			shadow.radius = 5;
			shadow.mapSize.set( 1024, 1024 );
			light.shadow = shadow;

			const json = light.toJSON();
			const newLight = new ObjectLoader().parse( json );

			expect( newLight.shadow ).toSmartEqual( light.shadow );

		} );

	} );

} );

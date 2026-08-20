import { describe, test, expect } from 'vitest';
import { SpotLightShadow } from '@src/lights/SpotLightShadow.js';
import { LightShadow } from '@src/lights/LightShadow.js';
import { SpotLight } from '@src/lights/SpotLight.js';
import { ObjectLoader } from '@src/loaders/ObjectLoader.js';

describe( 'Lights', () => {

	describe( 'SpotLightShadow', () => {

		test( 'Extending', () => {

			const object = new SpotLightShadow();
			expect( object instanceof LightShadow ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new SpotLightShadow();
			expect( object ).toBeTruthy();

		} );

		test( 'isSpotLightShadow', () => {

			const object = new SpotLightShadow();
			expect( object.isSpotLightShadow ).toBeTruthy();

		} );

		test( 'clone/copy', () => {

			const a = new SpotLightShadow();
			const b = new SpotLightShadow();

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

		test( 'toJSON', () => {

			const light = new SpotLight();
			const shadow = new SpotLightShadow();

			shadow.bias = 10;
			shadow.radius = 5;
			shadow.mapSize.set( 128, 128 );
			light.shadow = shadow;

			const json = light.toJSON();
			const newLight = new ObjectLoader().parse( json );

			expect( newLight.shadow ).toSmartEqual( light.shadow );

		} );

	} );

} );

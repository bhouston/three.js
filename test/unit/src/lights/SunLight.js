import { describe, test, expect } from 'vitest';
import { SunLight } from '@src/lights/SunLight.js';
import { Light } from '@src/lights/Light.js';
import { ObjectLoader } from '@src/loaders/ObjectLoader.js';
import { Vector3 } from '@src/math/Vector3.js';

describe( 'Lights', () => {

	describe( 'SunLight', () => {

		test( 'Extending', () => {

			const object = new SunLight();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new SunLight();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new SunLight();
			expect( object.type === 'SunLight' ).toBeTruthy();

		} );

		test( 'shadow', () => {

			const object = new SunLight();
			expect( object.shadow.isSunLightShadow ).toBeTruthy();

		} );

		test( 'isSunLight', () => {

			const object = new SunLight();
			expect( object.isSunLight ).toBeTruthy();

		} );

		test( 'position', () => {

			const object = new SunLight();
			expect( object.position.distanceTo( new Vector3( 0, 1, 0 ) ) < 1e-12 ).toBeTruthy();

		} );

		test( 'toJSON', () => {

			const light = new SunLight( 0xffaa88, 2 );
			light.shadow.bias = 10;
			light.position.setFromSphericalCoords( 1, 0.5, 1.2 );
			light.updateMatrix();

			const json = light.toJSON();
			const newLight = new ObjectLoader().parse( json );

			expect( newLight.isSunLight ).toBeTruthy();
			expect( newLight.shadow.isSunLightShadow ).toBeTruthy();
			expect( newLight.position.distanceTo( light.position ) < 1e-6 ).toBeTruthy();
			expect( newLight.shadow ).toSmartEqual( light.shadow );

		} );

	} );

} );

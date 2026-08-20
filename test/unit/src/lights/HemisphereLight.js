import { describe, test, expect, beforeEach } from 'vitest';
import { HemisphereLight } from '@src/lights/HemisphereLight.js';
import { Light } from '@src/lights/Light.js';
import { runStdLightTests } from '@test-utils/light-tests.js';

describe( 'Lights', () => {

	describe( 'HemisphereLight', () => {

		let lights = undefined;
		beforeEach( () => {

			const parameters = {
				skyColor: 0x123456,
				groundColor: 0xabc012,
				intensity: 0.6
			};

			lights = [
				new HemisphereLight(),
				new HemisphereLight( parameters.skyColor ),
				new HemisphereLight( parameters.skyColor, parameters.groundColor ),
				new HemisphereLight( parameters.skyColor, parameters.groundColor, parameters.intensity ),
			];

		} );

		test( 'Extending', () => {

			const object = new HemisphereLight();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new HemisphereLight();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new HemisphereLight();
			expect( object.type === 'HemisphereLight' ).toBeTruthy();

		} );

		test( 'isHemisphereLight', () => {

			const object = new HemisphereLight();
			expect( object.isHemisphereLight ).toBeTruthy();

		} );

		test( 'Standard light tests', () => {

			runStdLightTests( lights );

		} );

	} );

} );

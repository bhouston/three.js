import { describe, test, expect, beforeEach } from 'vitest';
import { AmbientLight } from '@src/lights/AmbientLight.js';
import { Light } from '@src/lights/Light.js';
import { runStdLightTests } from '@test-utils/light-tests.js';

describe( 'Lights', () => {

	describe( 'AmbientLight', () => {

		let lights = undefined;
		beforeEach( () => {

			const parameters = {
				color: 0xaaaaaa,
				intensity: 0.5
			};

			lights = [
				new AmbientLight(),
				new AmbientLight( parameters.color ),
				new AmbientLight( parameters.color, parameters.intensity )
			];

		} );

		test( 'Extending', () => {

			const object = new AmbientLight();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new AmbientLight();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new AmbientLight();
			expect( object.type === 'AmbientLight' ).toBeTruthy();

		} );

		test( 'isAmbientLight', () => {

			const object = new AmbientLight();
			expect( object.isAmbientLight ).toBeTruthy();

		} );

		test( 'Standard light tests', () => {

			runStdLightTests( lights );

		} );

	} );

} );

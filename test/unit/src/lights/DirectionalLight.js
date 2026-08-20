import { describe, test, expect, beforeEach } from 'vitest';
import { DirectionalLight } from '@src/lights/DirectionalLight.js';
import { Light } from '@src/lights/Light.js';
import { runStdLightTests } from '@test-utils/light-tests.js';

describe( 'Lights', () => {

	describe( 'DirectionalLight', () => {

		let lights = undefined;
		beforeEach( () => {

			const parameters = {
				color: 0xaaaaaa,
				intensity: 0.8
			};

			lights = [
				new DirectionalLight(),
				new DirectionalLight( parameters.color ),
				new DirectionalLight( parameters.color, parameters.intensity )
			];

		} );

		test( 'Extending', () => {

			const object = new DirectionalLight();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new DirectionalLight();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new DirectionalLight();
			expect( object.type === 'DirectionalLight' ).toBeTruthy();

		} );

		test( 'isDirectionalLight', () => {

			const object = new DirectionalLight();
			expect( object.isDirectionalLight ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new DirectionalLight();
			object.dispose();

			// ensure calls dispose() on shadow

		} );

		test( 'Standard light tests', () => {

			runStdLightTests( lights );

		} );

	} );

} );

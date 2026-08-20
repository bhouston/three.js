import { describe, test, expect, beforeEach } from 'vitest';
import { SpotLight } from '@src/lights/SpotLight.js';
import { Light } from '@src/lights/Light.js';
import { runStdLightTests } from '@test-utils/light-tests.js';

describe( 'Lights', () => {

	describe( 'SpotLight', () => {

		let lights = undefined;
		beforeEach( () => {

			const parameters = {
				color: 0xaaaaaa,
				intensity: 0.5,
				distance: 100,
				angle: 0.8,
				penumbra: 8,
				decay: 2
			};

			lights = [
				new SpotLight( parameters.color ),
				new SpotLight( parameters.color, parameters.intensity ),
				new SpotLight( parameters.color, parameters.intensity, parameters.distance ),
				new SpotLight( parameters.color, parameters.intensity, parameters.distance, parameters.angle ),
				new SpotLight( parameters.color, parameters.intensity, parameters.distance, parameters.angle, parameters.penumbra ),
				new SpotLight( parameters.color, parameters.intensity, parameters.distance, parameters.angle, parameters.penumbra, parameters.decay ),
			];

		} );

		test( 'Extending', () => {

			const object = new SpotLight();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new SpotLight();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new SpotLight();
			expect( object.type === 'SpotLight' ).toBeTruthy();

		} );

		test( 'power', () => {

			const a = new SpotLight( 0xaaaaaa );

			a.intensity = 100;
			expect( a.power ).toNumEqual( 100 * Math.PI );

			a.intensity = 40;
			expect( a.power ).toNumEqual( 40 * Math.PI );

			a.power = 100;
			expect( a.intensity ).toNumEqual( 100 / Math.PI );

		} );

		test( 'isSpotLight', () => {

			const object = new SpotLight();
			expect( object.isSpotLight ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new SpotLight();
			object.dispose();

			// ensure calls dispose() on shadow

		} );

		test( 'Standard light tests', () => {

			runStdLightTests( lights );

		} );

	} );

} );

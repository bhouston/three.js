import { describe, test, expect, beforeEach } from 'vitest';
import { PointLight } from '@src/lights/PointLight.js';
import { Light } from '@src/lights/Light.js';
import { runStdLightTests } from '@test-utils/light-tests.js';

describe( 'Lights', () => {

	describe( 'PointLight', () => {

		let lights = undefined;
		beforeEach( () => {

			const parameters = {
				color: 0xaaaaaa,
				intensity: 0.5,
				distance: 100,
				decay: 2
			};

			lights = [
				new PointLight(),
				new PointLight( parameters.color ),
				new PointLight( parameters.color, parameters.intensity ),
				new PointLight( parameters.color, parameters.intensity, parameters.distance ),
				new PointLight( parameters.color, parameters.intensity, parameters.distance, parameters.decay )
			];

		} );

		test( 'Extending', () => {

			const object = new PointLight();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new PointLight();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new PointLight();
			expect( object.type === 'PointLight' ).toBeTruthy();

		} );

		test( 'power', () => {

			const a = new PointLight( 0xaaaaaa );

			a.intensity = 100;
			expect( a.power ).toNumEqual( 100 * Math.PI * 4 );

			a.intensity = 40;
			expect( a.power ).toNumEqual( 40 * Math.PI * 4 );

			a.power = 100;
			expect( a.intensity ).toNumEqual( 100 / ( 4 * Math.PI ) );

		} );

		test( 'isPointLight', () => {

			const object = new PointLight();
			expect( object.isPointLight ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new PointLight();
			object.dispose();

			// ensure calls dispose() on shadow

		} );

		test( 'Standard light tests', () => {

			runStdLightTests( lights );

		} );

	} );

} );

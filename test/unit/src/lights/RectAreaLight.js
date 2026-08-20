import { describe, test, expect, beforeEach } from 'vitest';
import { RectAreaLight } from '@src/lights/RectAreaLight.js';
import { Light } from '@src/lights/Light.js';
import { runStdLightTests } from '@test-utils/light-tests.js';

describe( 'Lights', () => {

	describe( 'RectAreaLight', () => {

		let lights = undefined;
		beforeEach( () => {

			const parameters = {
				color: 0xaaaaaa,
				intensity: 0.5,
				width: 100,
				height: 50
			};

			lights = [
				new RectAreaLight( parameters.color ),
				new RectAreaLight( parameters.color, parameters.intensity ),
				new RectAreaLight( parameters.color, parameters.intensity, parameters.width ),
				new RectAreaLight( parameters.color, parameters.intensity, parameters.width, parameters.height )
			];

		} );

		test( 'Extending', () => {

			const object = new RectAreaLight();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new RectAreaLight();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new RectAreaLight();
			expect( object.type === 'RectAreaLight' ).toBeTruthy();

		} );

		test( 'power', () => {

			const a = new RectAreaLight( 0xaaaaaa, 1, 10, 10 );
			let actual = undefined;
			let expected = undefined;

			a.intensity = 100;
			actual = a.power;
			expected = 100 * a.width * a.height * Math.PI;
			expect( actual ).toNumEqual( expected );

			a.intensity = 40;
			actual = a.power;
			expected = 40 * a.width * a.height * Math.PI;
			expect( actual ).toNumEqual( expected );

			a.power = 100;
			actual = a.intensity;
			expected = 100 / ( a.width * a.height * Math.PI );
			expect( actual ).toNumEqual( expected );

		} );

		test( 'isRectAreaLight', () => {

			const object = new RectAreaLight();
			expect( object.isRectAreaLight ).toBeTruthy();

		} );

		test( 'Standard light tests', () => {

			runStdLightTests( lights );

		} );

	} );

} );

import { describe, test, expect, beforeEach } from 'vitest';
import { Light } from '@src/lights/Light.js';
import { Object3D } from '@src/core/Object3D.js';
import { runStdLightTests } from '@test-utils/light-tests.js';

describe( 'Lights', () => {

	describe( 'Light', () => {

		let lights = undefined;
		beforeEach( () => {

			const parameters = {
				color: 0xaaaaaa,
				intensity: 0.5
			};

			lights = [
				new Light(),
				new Light( parameters.color ),
				new Light( parameters.color, parameters.intensity )
			];

		} );

		test( 'Extending', () => {

			const object = new Light();
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new Light();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new Light();
			expect( object.type === 'Light' ).toBeTruthy();

		} );

		test( 'isLight', () => {

			const object = new Light();
			expect( object.isLight ).toBeTruthy();

		} );

		test( 'dispose', () => {

			// empty, test exists
			const object = new Light();
			object.dispose();

		} );

		test( 'Standard light tests', () => {

			runStdLightTests( lights );

		} );

	} );

} );

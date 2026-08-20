import { describe, test, expect } from 'vitest';

import { DirectionalLightHelper } from '@src/helpers/DirectionalLightHelper.js';

import { Object3D } from '@src/core/Object3D.js';
import { DirectionalLight } from '@src/lights/DirectionalLight.js';

describe( 'Helpers', () => {

	describe( 'DirectionalLightHelper', () => {

		const parameters = {
			size: 1,
			color: 0xaaaaaa,
			intensity: 0.8
		};

		test( 'Extending', () => {

			const light = new DirectionalLight( parameters.color );
			const object = new DirectionalLightHelper( light, parameters.size, parameters.color );
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const light = new DirectionalLight( parameters.color );
			const object = new DirectionalLightHelper( light, parameters.size, parameters.color );
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const light = new DirectionalLight( parameters.color );
			const object = new DirectionalLightHelper( light, parameters.size, parameters.color );
			expect( object.type === 'DirectionalLightHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const light = new DirectionalLight( parameters.color );
			const object = new DirectionalLightHelper( light, parameters.size, parameters.color );
			object.dispose();

		} );

	} );

} );

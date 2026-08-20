import { describe, test, expect } from 'vitest';

import { PointLightHelper } from '@src/helpers/PointLightHelper.js';

import { Mesh } from '@src/objects/Mesh.js';
import { PointLight } from '@src/lights/PointLight.js';

describe( 'Helpers', () => {

	describe( 'PointLightHelper', () => {

		const parameters = {
			sphereSize: 1,
			color: 0xaaaaaa,
			intensity: 0.5,
			distance: 100,
			decay: 2
		};

		test( 'Extending', () => {

			const light = new PointLight( parameters.color );
			const object = new PointLightHelper( light, parameters.sphereSize, parameters.color );
			expect( object instanceof Mesh ).toBe( true );

		} );

		test( 'Instancing', () => {

			const light = new PointLight( parameters.color );
			const object = new PointLightHelper( light, parameters.sphereSize, parameters.color );
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const light = new PointLight( parameters.color );
			const object = new PointLightHelper( light, parameters.sphereSize, parameters.color );
			expect( object.type === 'PointLightHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const light = new PointLight( parameters.color );
			const object = new PointLightHelper( light, parameters.sphereSize, parameters.color );
			object.dispose();

		} );

	} );

} );

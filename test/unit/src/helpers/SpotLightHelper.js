import { describe, test, expect } from 'vitest';

import { SpotLightHelper } from '@src/helpers/SpotLightHelper.js';

import { Object3D } from '@src/core/Object3D.js';
import { SpotLight } from '@src/lights/SpotLight.js';

describe( 'Helpers', () => {

	describe( 'SpotLightHelper', () => {

		const parameters = {
			color: 0xaaaaaa,
			intensity: 0.5,
			distance: 100,
			angle: 0.8,
			penumbra: 8,
			decay: 2
		};

		test( 'Extending', () => {

			const light = new SpotLight( parameters.color );
			const object = new SpotLightHelper( light, parameters.color );
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const light = new SpotLight( parameters.color );
			const object = new SpotLightHelper( light, parameters.color );
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const light = new SpotLight( parameters.color );
			const object = new SpotLightHelper( light, parameters.color );
			expect( object.type === 'SpotLightHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const light = new SpotLight( parameters.color );
			const object = new SpotLightHelper( light, parameters.color );
			object.dispose();

		} );

	} );

} );

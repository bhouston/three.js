import { describe, test, expect } from 'vitest';

import { HemisphereLightHelper } from '@src/helpers/HemisphereLightHelper.js';

import { Object3D } from '@src/core/Object3D.js';
import { HemisphereLight } from '@src/lights/HemisphereLight.js';

describe( 'Helpers', () => {

	describe( 'HemisphereLightHelper', () => {

		const parameters = {
			size: 1,
			color: 0xabc012,
			skyColor: 0x123456,
			groundColor: 0xabc012,
			intensity: 0.6
		};

		test( 'Extending', () => {

			const light = new HemisphereLight( parameters.skyColor );
			const object = new HemisphereLightHelper( light, parameters.size, parameters.color );
			expect( object instanceof Object3D ).toBe( true );

		} );

		test( 'Instancing', () => {

			const light = new HemisphereLight( parameters.skyColor );
			const object = new HemisphereLightHelper( light, parameters.size, parameters.color );
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const light = new HemisphereLight( parameters.skyColor );
			const object = new HemisphereLightHelper( light, parameters.size, parameters.color );
			expect( object.type === 'HemisphereLightHelper' ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const light = new HemisphereLight( parameters.skyColor );
			const object = new HemisphereLightHelper( light, parameters.size, parameters.color );
			object.dispose();

		} );

	} );

} );

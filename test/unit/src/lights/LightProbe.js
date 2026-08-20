import { describe, test, expect } from 'vitest';
import { LightProbe } from '@src/lights/LightProbe.js';
import { Light } from '@src/lights/Light.js';

describe( 'Lights', () => {

	describe( 'LightProbe', () => {

		test( 'Extending', () => {

			const object = new LightProbe();
			expect( object instanceof Light ).toBe( true );

		} );

		test( 'isLightProbe', () => {

			const object = new LightProbe();
			expect( object.isLightProbe ).toBeTruthy();

		} );

	} );

} );

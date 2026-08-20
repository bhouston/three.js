import { describe, test, expect } from 'vitest';

import { UniformsGroup } from '@src/core/UniformsGroup.js';
import { EventDispatcher } from '@src/core/EventDispatcher.js';

describe( 'Core', () => {

	describe( 'UniformsGroup', () => {

		test( 'Extending', () => {

			const object = new UniformsGroup();
			expect( object instanceof EventDispatcher ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new UniformsGroup();
			expect( object ).toBeTruthy();

		} );

		test( 'isUniformsGroup', () => {

			const object = new UniformsGroup();
			expect( object.isUniformsGroup ).toBeTruthy();

		} );

		test( 'dispose', () => {

			const object = new UniformsGroup();
			object.dispose();

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { Object3D } from '@src/core/Object3D.js';
import { Line } from '@src/objects/Line.js';
import { LineLoop } from '@src/objects/LineLoop.js';

describe( 'Objects', () => {

	describe( 'LineLoop', () => {

		test( 'Extending', () => {

			const lineLoop = new LineLoop();

			expect( lineLoop instanceof Object3D ).toBe( true );
			expect( lineLoop instanceof Line ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new LineLoop();
			expect( object ).toBeTruthy();

		} );

		test( 'type', () => {

			const object = new LineLoop();
			expect( object.type === 'LineLoop' ).toBeTruthy();

		} );

		test( 'isLineLoop', () => {

			const object = new LineLoop();
			expect( object.isLineLoop ).toBeTruthy();

		} );

	} );

} );

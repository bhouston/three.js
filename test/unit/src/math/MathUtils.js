import { describe, test, expect } from 'vitest';
import * as MathUtils from '@src/math/MathUtils.js';

describe( 'Maths', () => {

	describe( 'Math', () => {

		test( 'generateUUID', () => {

			const a = MathUtils.generateUUID();
			const regex = /[A-Z0-9]{8}-[A-Z0-9]{4}-4[A-Z0-9]{3}-[A-Z0-9]{4}-[A-Z0-9]{12}/i;
			// note the fixed '4' here ----------^

			expect( regex.test( a ) ).toBeTruthy();

		} );

		test( 'clamp', () => {

			expect( MathUtils.clamp( 0.5, 0, 1 ) ).toBe( 0.5 );
			expect( MathUtils.clamp( 0, 0, 1 ) ).toBe( 0 );
			expect( MathUtils.clamp( - 0.1, 0, 1 ) ).toBe( 0 );
			expect( MathUtils.clamp( 1.1, 0, 1 ) ).toBe( 1 );

		} );

		test( 'euclideanModulo', () => {

			expect( isNaN( MathUtils.euclideanModulo( 6, 0 ) ) ).toBeTruthy();
			expect( MathUtils.euclideanModulo( 6, 1 ) ).toBe( 0 );
			expect( MathUtils.euclideanModulo( 6, 2 ) ).toBe( 0 );
			expect( MathUtils.euclideanModulo( 6, 5 ) ).toBe( 1 );
			expect( MathUtils.euclideanModulo( 6, 6 ) ).toBe( 0 );
			expect( MathUtils.euclideanModulo( 6, 7 ) ).toBe( 6 );

		} );

		test( 'mapLinear', () => {

			expect( MathUtils.mapLinear( 0.5, 0, 1, 0, 10 ) ).toBe( 5 );
			expect( MathUtils.mapLinear( 0.0, 0, 1, 0, 10 ) ).toBe( 0 );
			expect( MathUtils.mapLinear( 1.0, 0, 1, 0, 10 ) ).toBe( 10 );

		} );

		test( 'inverseLerp', () => {

			expect( MathUtils.inverseLerp( 1, 2, 1.5 ) ).toBe( 0.5 );
			expect( MathUtils.inverseLerp( 1, 2, 2 ) ).toBe( 1 );
			expect( MathUtils.inverseLerp( 1, 2, 1 ) ).toBe( 0 );
			expect( MathUtils.inverseLerp( 1, 1, 1 ) ).toBe( 0 );

		} );

		test( 'lerp', () => {

			expect( MathUtils.lerp( 1, 2, 0 ) ).toBe( 1 );
			expect( MathUtils.lerp( 1, 2, 1 ) ).toBe( 2 );
			expect( MathUtils.lerp( 1, 2, 0.4 ) ).toBe( 1.4 );

		} );

		test( 'damp', () => {

			expect( MathUtils.damp( 1, 2, 0, 0.016 ) ).toBe( 1 );
			expect( MathUtils.damp( 1, 2, 10, 0.016 ) ).toBe( 1.1478562110337887 );

		} );

		test( 'pingpong', () => {

			expect( MathUtils.pingpong( 2.5 ) ).toBe( 0.5 );
			expect( MathUtils.pingpong( 2.5, 2 ) ).toBe( 1.5 );
			expect( MathUtils.pingpong( - 1.5 ) ).toBe( 0.5 );

		} );

		test( 'smoothstep', () => {

			expect( MathUtils.smoothstep( - 1, 0, 2 ) ).toBe( 0 );
			expect( MathUtils.smoothstep( 0, 0, 2 ) ).toBe( 0 );
			expect( MathUtils.smoothstep( 0.5, 0, 2 ) ).toBe( 0.15625 );
			expect( MathUtils.smoothstep( 1, 0, 2 ) ).toBe( 0.5 );
			expect( MathUtils.smoothstep( 1.5, 0, 2 ) ).toBe( 0.84375 );
			expect( MathUtils.smoothstep( 2, 0, 2 ) ).toBe( 1 );
			expect( MathUtils.smoothstep( 3, 0, 2 ) ).toBe( 1 );

		} );

		test( 'smootherstep', () => {

			expect( MathUtils.smootherstep( - 1, 0, 2 ) ).toBe( 0 );
			expect( MathUtils.smootherstep( 0, 0, 2 ) ).toBe( 0 );
			expect( MathUtils.smootherstep( 0.5, 0, 2 ) ).toBe( 0.103515625 );
			expect( MathUtils.smootherstep( 1, 0, 2 ) ).toBe( 0.5 );
			expect( MathUtils.smootherstep( 1.5, 0, 2 ) ).toBe( 0.896484375 );
			expect( MathUtils.smootherstep( 2, 0, 2 ) ).toBe( 1 );
			expect( MathUtils.smootherstep( 3, 0, 2 ) ).toBe( 1 );

		} );

		test( 'randInt', () => {

			const low = 1, high = 3;
			const a = MathUtils.randInt( low, high );

			expect( a >= low ).toBeTruthy();
			expect( a <= high ).toBeTruthy();

		} );

		test( 'randFloat', () => {

			const low = 1, high = 3;
			const a = MathUtils.randFloat( low, high );

			expect( a >= low ).toBeTruthy();
			expect( a <= high ).toBeTruthy();

		} );

		test( 'randFloatSpread', () => {

			const a = MathUtils.randFloatSpread( 3 );

			expect( a > - 3 / 2 ).toBeTruthy();
			expect( a < 3 / 2 ).toBeTruthy();

		} );

		test( 'degToRad', () => {

			expect( MathUtils.degToRad( 0 ) ).toBe( 0 );
			expect( MathUtils.degToRad( 90 ) ).toBe( Math.PI / 2 );
			expect( MathUtils.degToRad( 180 ) ).toBe( Math.PI );
			expect( MathUtils.degToRad( 360 ) ).toBe( Math.PI * 2 );

		} );

		test( 'radToDeg', () => {

			expect( MathUtils.radToDeg( 0 ) ).toBe( 0 );
			expect( MathUtils.radToDeg( Math.PI / 2 ) ).toBe( 90 );
			expect( MathUtils.radToDeg( Math.PI ) ).toBe( 180 );
			expect( MathUtils.radToDeg( Math.PI * 2 ) ).toBe( 360 );

		} );

		test( 'isPowerOfTwo', () => {

			expect( MathUtils.isPowerOfTwo( 0 ) ).toBe( false );
			expect( MathUtils.isPowerOfTwo( 1 ) ).toBe( true );
			expect( MathUtils.isPowerOfTwo( 2 ) ).toBe( true );
			expect( MathUtils.isPowerOfTwo( 3 ) ).toBe( false );
			expect( MathUtils.isPowerOfTwo( 4 ) ).toBe( true );

		} );

		test( 'ceilPowerOfTwo', () => {

			expect( MathUtils.ceilPowerOfTwo( 1 ) ).toBe( 1 );
			expect( MathUtils.ceilPowerOfTwo( 3 ) ).toBe( 4 );
			expect( MathUtils.ceilPowerOfTwo( 4 ) ).toBe( 4 );

		} );

		test( 'floorPowerOfTwo', () => {

			expect( MathUtils.floorPowerOfTwo( 1 ) ).toBe( 1 );
			expect( MathUtils.floorPowerOfTwo( 3 ) ).toBe( 2 );
			expect( MathUtils.floorPowerOfTwo( 4 ) ).toBe( 4 );

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { AnimationClip } from '@src/animation/AnimationClip.js';

describe( 'Animation', () => {

	describe( 'AnimationClip', () => {

		test( 'Instancing', () => {

			const clip = new AnimationClip( 'clip1', 1000, [ {} ] );
			expect( clip ).toBeTruthy();

		} );

		test( 'name', () => {

			const clip = new AnimationClip( 'clip1', 1000, [ {} ] );
			expect( clip.name === 'clip1' ).toBe( true );

		} );

	} );

} );

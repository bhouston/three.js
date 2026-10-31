import { describe, test } from 'vitest';
import { assert } from '../../utils/assertions.js';

import { AnimationClip } from '../../../../src/animation/AnimationClip.js';

describe( 'Animation', () => {

	describe( 'AnimationClip', () => {

		// INSTANCING
		test( 'Instancing', () => {

			const clip = new AnimationClip( 'clip1', 1000, [ {} ] );
			assert.ok( clip, 'AnimationClip can be instantiated' );

		} );

		// PROPERTIES
		test( 'name', () => {

			const clip = new AnimationClip( 'clip1', 1000, [ {} ] );
			assert.strictEqual(
				clip.name === 'clip1', true,
				'AnimationClip can be named'
			);

		} );

	} );

} );

import { describe, test, expect } from 'vitest';
import { AnimationMixer } from '@src/animation/AnimationMixer.js';

import { EventDispatcher } from '@src/core/EventDispatcher.js';
import { AnimationClip } from '@src/animation/AnimationClip.js';
import { VectorKeyframeTrack } from '@src/animation/tracks/VectorKeyframeTrack.js';
import { Object3D } from '@src/core/Object3D.js';
import { zero3, one3, two3 } from '@test-utils/math-constants.js';

function getClips( pos1, pos2, scale1, scale2, dur ) {

	const clips = [];

	let track = new VectorKeyframeTrack( '.scale', [ 0, dur ], [ scale1.x, scale1.y, scale1.z, scale2.x, scale2.y, scale2.z ] );
	clips.push( new AnimationClip( 'scale', dur, [ track ] ) );

	track = new VectorKeyframeTrack( '.position', [ 0, dur ], [ pos1.x, pos1.y, pos1.z, pos2.x, pos2.y, pos2.z ] );
	clips.push( new AnimationClip( 'position', dur, [ track ] ) );

	return clips;

}

describe( 'Animation', () => {

	describe( 'AnimationMixer', () => {

		test( 'Extending', () => {

			const object = new AnimationMixer();
			expect( object instanceof EventDispatcher ).toBe( true );

		} );

		test( 'Instancing', () => {

			const object = new AnimationMixer();
			expect( object ).toBeTruthy();

		} );

		test( 'stopAllAction', () => {

			const obj = new Object3D();
			const animMixer = new AnimationMixer( obj );
			const clips = getClips( zero3, one3, two3, one3, 1 );
			const actionA = animMixer.clipAction( clips[ 0 ] );
			const actionB = animMixer.clipAction( clips[ 1 ] );

			actionA.play();
			actionB.play();
			animMixer.update( 0.1 );
			animMixer.stopAllAction();

			expect(
				! actionA.isRunning() &&
				! actionB.isRunning()
			).toBeTruthy();
			expect(
				obj.position.x == 0 &&
				obj.position.y == 0 &&
				obj.position.z == 0
			).toBeTruthy();
			expect(
				obj.scale.x == 1 &&
				obj.scale.y == 1 &&
				obj.scale.z == 1
			).toBeTruthy();

		} );

		test( 'getRoot', () => {

			const obj = new Object3D();
			const animMixer = new AnimationMixer( obj );
			expect( obj ).toBe( animMixer.getRoot() );

		} );

	} );

} );

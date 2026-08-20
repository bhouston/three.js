import { describe, test, expect } from 'vitest';
import { AnimationAction } from '@src/animation/AnimationAction.js';

import { AnimationMixer } from '@src/animation/AnimationMixer.js';
import { AnimationClip } from '@src/animation/AnimationClip.js';
import { NumberKeyframeTrack } from '@src/animation/tracks/NumberKeyframeTrack.js';
import { Object3D } from '@src/core/Object3D.js';
import { LoopOnce, LoopRepeat, LoopPingPong } from '@src/constants.js';

function createAnimation() {

	const root = new Object3D();
	const mixer = new AnimationMixer( root );
	const track = new NumberKeyframeTrack( '.rotation[x]', [ 0, 1000 ], [ 0, 360 ] );
	const clip = new AnimationClip( 'clip1', 1000, [ track ] );

	const animationAction = mixer.clipAction( clip );
	return {
		root: root,
		mixer: mixer,
		track: track,
		clip: clip,
		animationAction: animationAction
	};

}

function createTwoAnimations() {

	const root = new Object3D();
	const mixer = new AnimationMixer( root );
	const track = new NumberKeyframeTrack( '.rotation[x]', [ 0, 1000 ], [ 0, 360 ] );
	const clip = new AnimationClip( 'clip1', 1000, [ track ] );
	const animationAction = mixer.clipAction( clip );

	const track2 = new NumberKeyframeTrack( '.rotation[y]', [ 0, 1000 ], [ 0, 360 ] );
	const clip2 = new AnimationClip( 'clip2', 1000, [ track ] );
	const animationAction2 = mixer.clipAction( clip2 );

	return {
		root: root,
		mixer: mixer,
		track: track,
		clip: clip,
		animationAction: animationAction,
		track2: track2,
		clip2: clip2,
		animationAction2: animationAction2
	};

}

describe( 'Animation', () => {

	describe( 'AnimationAction', () => {

		test( 'Instancing', () => {

			const mixer = new AnimationMixer();
			const clip = new AnimationClip( 'nonname', - 1, [] );

			const animationAction = new AnimationAction( mixer, clip );
			expect( animationAction ).toBeTruthy();

		} );

		test( 'play', () => {

			const { mixer, animationAction } = createAnimation();
			const animationAction2 = animationAction.play();
			expect( animationAction ).toBe( animationAction2 );

			const UserException = function () {

				this.message = 'AnimationMixer must activate AnimationAction on play.';

			};

			mixer._activateAction = function ( action ) {

				if ( action === animationAction ) {

					throw new UserException();

				}

			};

			expect( () => {

				animationAction.play();

			} ).toThrow( new UserException() );

		} );

		test( 'stop', () => {

			const { mixer, animationAction } = createAnimation();
			const animationAction2 = animationAction.stop();
			expect( animationAction ).toBe( animationAction2 );

			const UserException = function () {

				this.message = 'AnimationMixer must deactivate AnimationAction on stop.';

			};

			mixer._deactivateAction = function ( action ) {

				if ( action === animationAction ) {

					throw new UserException();

				}

			};

			expect( () => {

				animationAction.stop();

			} ).toThrow( new UserException() );

		} );

		test( 'reset', () => {

			const { animationAction } = createAnimation();
			const animationAction2 = animationAction.stop();
			expect( animationAction ).toBe( animationAction2 );
			expect( animationAction2.paused ).toBe( false );
			expect( animationAction2.enabled ).toBe( true );
			expect( animationAction2.time ).toBe( 0 );
			expect( animationAction2._loopCount ).toBe( - 1 );
			expect( animationAction2._startTime ).toBe( null );

		} );

		test( 'isRunning', () => {

			const { animationAction } = createAnimation();
			expect( animationAction.isRunning() ).toBeFalsy();
			animationAction.play();
			expect( animationAction.isRunning() ).toBeTruthy();
			animationAction.stop();
			expect( animationAction.isRunning() ).toBeFalsy();
			animationAction.play();
			animationAction.paused = true;
			expect( animationAction.isRunning() ).toBeFalsy();
			animationAction.paused = false;
			animationAction.enabled = false;
			expect( animationAction.isRunning() ).toBeFalsy();
			animationAction.enabled = true;
			expect( animationAction.isRunning() ).toBeTruthy();

		} );

		test( 'isScheduled', () => {

			const { mixer, animationAction } = createAnimation();
			expect( animationAction.isScheduled() ).toBeFalsy();
			animationAction.play();
			expect( animationAction.isScheduled() ).toBeTruthy();
			mixer.update( 1 );
			expect( animationAction.isScheduled() ).toBeTruthy();
			animationAction.stop();
			expect( animationAction.isScheduled() ).toBeFalsy();

		} );

		test( 'startAt', () => {

			const { mixer, animationAction } = createAnimation();
			animationAction.startAt( 2 );
			animationAction.play();
			expect( animationAction.isRunning() ).toBeFalsy();
			expect( animationAction.isScheduled() ).toBeTruthy();
			mixer.update( 1 );
			expect( animationAction.isRunning() ).toBeFalsy();
			expect( animationAction.isScheduled() ).toBeTruthy();
			mixer.update( 1 );
			expect( animationAction.isRunning() ).toBeTruthy();
			expect( animationAction.isScheduled() ).toBeTruthy();
			animationAction.stop();
			expect( animationAction.isRunning() ).toBeFalsy();
			expect( animationAction.isScheduled() ).toBeFalsy();

		} );

		test( 'setLoop LoopOnce', () => {

			const { mixer, animationAction } = createAnimation();
			animationAction.setLoop( LoopOnce );
			animationAction.play();
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 500 );
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 500 );
			expect( animationAction.isRunning() ).toBeFalsy();
			mixer.update( 500 );
			expect( animationAction.isRunning() ).toBeFalsy();

		} );

		test( 'setLoop LoopRepeat', () => {

			const { root, mixer, animationAction } = createAnimation();
			animationAction.setLoop( LoopRepeat, 3 );
			animationAction.play();
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 750 );
			expect( root.rotation.x ).toBe( 270 );
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 270 );
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 270 );
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 0 );
			expect( animationAction.isRunning() ).toBeFalsy();

		} );

		test( 'setLoop LoopPingPong', () => {

			const { root, mixer, animationAction } = createAnimation();
			animationAction.setLoop( LoopPingPong, 3 );
			animationAction.play();
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 750 );
			expect( root.rotation.x ).toBe( 270 );
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 90 );
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 270 );
			expect( animationAction.isRunning() ).toBeTruthy();
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 0 );
			expect( animationAction.isRunning() ).toBeFalsy();

		} );

		test( 'setEffectiveWeight', () => {

			const { animationAction } = createAnimation();
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			animationAction.setEffectiveWeight( 0.3 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.3 );

		} );

		test( 'setEffectiveWeight - disabled', () => {

			const { animationAction } = createAnimation();
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			animationAction.enabled = false;
			animationAction.setEffectiveWeight( 0.3 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0 );

		} );

		test( 'setEffectiveWeight - over duration', () => {

			const { root, mixer, animationAction } = createAnimation();
			animationAction.setEffectiveWeight( 0.5 );
			animationAction.play();
			mixer.update( 500 );
			expect( root.rotation.x ).toBe( 90 );
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 90 );

		} );

		test( 'getEffectiveWeight', () => {

			let { animationAction } = createAnimation();
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			animationAction.setEffectiveWeight( 0.3 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.3 );

			( { animationAction } = createAnimation() );
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			animationAction.enabled = false;
			animationAction.setEffectiveWeight( 0.3 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0 );

		} );

		test( 'fadeIn', () => {

			const { mixer, animationAction } = createAnimation();
			animationAction.fadeIn( 1000 );
			animationAction.play();
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.25 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.5 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.75 );
			mixer.update( 500 );
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );

		} );

		test( 'fadeOut', () => {

			const { mixer, animationAction } = createAnimation();
			animationAction.fadeOut( 1000 );
			animationAction.play();
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.75 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.5 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.25 );
			mixer.update( 500 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0 );

		} );

		test( 'crossFadeFrom', () => {

			const { mixer, animationAction, animationAction2 } = createTwoAnimations();
			animationAction.crossFadeFrom( animationAction2, 1000, false );
			animationAction.play();
			animationAction2.play();
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 1 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.25 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0.75 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.5 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0.5 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.75 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0.25 );
			mixer.update( 500 );
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0 );

		} );

		test( 'crossFadeTo', () => {

			const { mixer, animationAction, animationAction2 } = createTwoAnimations();
			animationAction2.crossFadeTo( animationAction, 1000, false );
			animationAction.play();
			animationAction2.play();
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 1 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.25 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0.75 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.5 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0.5 );
			mixer.update( 250 );
			expect( animationAction.getEffectiveWeight() ).toBe( 0.75 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0.25 );
			mixer.update( 500 );
			expect( animationAction.getEffectiveWeight() ).toBe( 1 );
			expect( animationAction2.getEffectiveWeight() ).toBe( 0 );

		} );

		test( 'getMixer', () => {

			const { mixer, animationAction } = createAnimation();
			const mixer2 = animationAction.getMixer();
			expect( mixer ).toBe( mixer2 );

		} );

		test( 'getClip', () => {

			const { clip, animationAction } = createAnimation();
			const clip2 = animationAction.getClip();
			expect( clip ).toBe( clip2 );

		} );

		test( 'getRoot', () => {

			const { root, animationAction } = createAnimation();
			const root2 = animationAction.getRoot();
			expect( root ).toBe( root2 );

		} );

		test( 'StartAt when already executed once', () => {

			const root = new Object3D();
			const mixer = new AnimationMixer( root );
			const track = new NumberKeyframeTrack( '.rotation[x]', [ 0, 750 ], [ 0, 270 ] );
			const clip = new AnimationClip( 'clip1', 750, [ track ] );

			const animationAction = mixer.clipAction( clip );
			animationAction.setLoop( LoopOnce );
			animationAction.clampWhenFinished = true;
			animationAction.play();
			mixer.addEventListener( 'finished', () => {

				animationAction.timeScale *= - 1;
				animationAction.paused = false;
				animationAction.startAt( mixer.time + 2000 ).play();

			} );

			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 90 );
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 180 );
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 270 );
			//first loop done
			mixer.update( 2000 );
			// startAt Done
			expect( root.rotation.x ).toBe( 270 );
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 180 );
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 90 );
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 0 );
			mixer.update( 1 );
			expect( root.rotation.x ).toBe( 0 );
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 0 );
			mixer.update( 1000 );
			expect( root.rotation.x ).toBe( 0 );
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 90 );
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 180 );

		} );

		test( 'LoopRepeat with timeScale reversal during first loop', () => {

			// Regression test for #19151
			const root = new Object3D();
			const mixer = new AnimationMixer( root );
			const track = new NumberKeyframeTrack( '.rotation[x]', [ 0, 1000 ], [ 0, 360 ] );
			const clip = new AnimationClip( 'clip1', 1000, [ track ] );

			const animationAction = mixer.clipAction( clip );
			animationAction.setLoop( LoopRepeat, Infinity );
			animationAction.play();

			// Advance partway into the first loop
			mixer.update( 500 );
			expect( root.rotation.x ).toBe( 180 );

			// Reverse timeScale
			mixer.timeScale = - 1;

			// Step backward — should smoothly reverse, not jump
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 90 );

			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 0 );

		} );

		test( 'LoopPingPong with timeScale reversal during first loop', () => {

			// Regression test for #19151
			const root = new Object3D();
			const mixer = new AnimationMixer( root );
			const track = new NumberKeyframeTrack( '.rotation[x]', [ 0, 1000 ], [ 0, 360 ] );
			const clip = new AnimationClip( 'clip1', 1000, [ track ] );

			const animationAction = mixer.clipAction( clip );
			animationAction.setLoop( LoopPingPong, Infinity );
			animationAction.play();

			// Advance partway into the first loop
			mixer.update( 500 );
			expect( root.rotation.x ).toBe( 180 );

			// Reverse timeScale
			mixer.timeScale = - 1;

			// Step backward — should smoothly reverse, not jump
			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 90 );

			mixer.update( 250 );
			expect( root.rotation.x ).toBe( 0 );

		} );

	} );

} );

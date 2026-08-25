import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createRandom, getLearningRate, yieldToBrowser } from '../../../../examples/jsm/ntc/training/NTCTrainingUtils.js';

describe( 'Addons > Neural > NeuralTrainingUtils', () => {

	describe( 'getLearningRate', () => {

		it( 'starts at the full learningRate on iteration 0', () => {

			const rate = getLearningRate( { learningRate: 0.1, iterations: 100, cosineAnnealingScale: 0.1 }, 0 );

			expect( rate ).toBeCloseTo( 0.1, 10 );

		} );

		it( 'decays to learningRate * cosineAnnealingScale at the final iteration', () => {

			const options = { learningRate: 0.1, iterations: 100, cosineAnnealingScale: 0.1 };
			const rate = getLearningRate( options, options.iterations - 1 );

			expect( rate ).toBeCloseTo( 0.1 * 0.1, 10 );

		} );

		it( 'is monotonically non-increasing across the training run', () => {

			const options = { learningRate: 1, iterations: 50, cosineAnnealingScale: 0 };
			let previous = Infinity;

			for ( let i = 0; i < options.iterations; i ++ ) {

				const rate = getLearningRate( options, i );
				expect( rate ).toBeLessThanOrEqual( previous + 1e-12 );
				previous = rate;

			}

		} );

		it( 'clamps t at 1 for an iteration index past the end (does not go negative or overshoot)', () => {

			const options = { learningRate: 1, iterations: 10, cosineAnnealingScale: 0.2 };
			const atEnd = getLearningRate( options, 9 );
			const pastEnd = getLearningRate( options, 500 );

			expect( pastEnd ).toBeCloseTo( atEnd, 10 );
			expect( pastEnd ).toBeCloseTo( 0.2, 10 );

		} );

		it( 'treats iterations = 1 without dividing by zero', () => {

			const rate = getLearningRate( { learningRate: 1, iterations: 1, cosineAnnealingScale: 0.1 }, 0 );

			expect( Number.isFinite( rate ) ).toBe( true );

		} );

		it( 'a cosineAnnealingScale of 1 holds the rate constant across the run', () => {

			const options = { learningRate: 0.05, iterations: 20, cosineAnnealingScale: 1 };

			for ( const iteration of [ 0, 5, 10, 19 ] ) {

				expect( getLearningRate( options, iteration ) ).toBeCloseTo( 0.05, 10 );

			}

		} );

	} );

	describe( 'createRandom', () => {

		it( 'produces values in [0, 1)', () => {

			const random = createRandom( 12345 );

			for ( let i = 0; i < 200; i ++ ) {

				const value = random();
				expect( value ).toBeGreaterThanOrEqual( 0 );
				expect( value ).toBeLessThan( 1 );

			}

		} );

		it( 'is fully deterministic for a given seed', () => {

			const a = createRandom( 42 );
			const b = createRandom( 42 );

			const sequenceA = Array.from( { length: 20 }, () => a() );
			const sequenceB = Array.from( { length: 20 }, () => b() );

			expect( sequenceA ).toEqual( sequenceB );

		} );

		it( 'produces different sequences for different seeds', () => {

			const a = createRandom( 1 );
			const b = createRandom( 2 );

			const sequenceA = Array.from( { length: 10 }, () => a() );
			const sequenceB = Array.from( { length: 10 }, () => b() );

			expect( sequenceA ).not.toEqual( sequenceB );

		} );

		it( 'does not repeat the same value on every call (not a degenerate constant generator)', () => {

			const random = createRandom( 7 );
			const values = new Set( Array.from( { length: 20 }, () => random() ) );

			expect( values.size ).toBeGreaterThan( 1 );

		} );

		it( 'accepts a zero seed without producing NaN', () => {

			const random = createRandom( 0 );
			const value = random();

			expect( Number.isFinite( value ) ).toBe( true );

		} );

		it( 'wraps a negative or out-of-int32-range seed via >>> 0 instead of throwing', () => {

			const random = createRandom( - 1 );
			const value = random();

			expect( Number.isFinite( value ) ).toBe( true );
			expect( value ).toBeGreaterThanOrEqual( 0 );
			expect( value ).toBeLessThan( 1 );

		} );

	} );

	describe( 'yieldToBrowser', () => {

		beforeEach( () => {

			vi.useFakeTimers();

		} );

		afterEach( () => {

			vi.useRealTimers();

		} );

		it( 'resolves via setTimeout when requestAnimationFrame is unavailable (Node environment)', async () => {

			expect( typeof globalThis.requestAnimationFrame ).toBe( 'undefined' );

			let resolved = false;
			yieldToBrowser().then( () => {

				resolved = true;

			} );

			expect( resolved ).toBe( false );

			await vi.runAllTimersAsync();

			expect( resolved ).toBe( true );

		} );

		it( 'prefers requestAnimationFrame when it is available', async () => {

			let rafCallback;
			globalThis.requestAnimationFrame = ( callback ) => {

				rafCallback = callback;
				return 1;

			};

			try {

				let resolved = false;
				yieldToBrowser().then( () => {

					resolved = true;

				} );

				await Promise.resolve(); // let the yieldToBrowser body run synchronously to registration
				expect( resolved ).toBe( false );
				expect( typeof rafCallback ).toBe( 'function' );

				rafCallback();
				await vi.runAllTimersAsync();

				expect( resolved ).toBe( true );

			} finally {

				delete globalThis.requestAnimationFrame;

			}

		} );

	} );

} );

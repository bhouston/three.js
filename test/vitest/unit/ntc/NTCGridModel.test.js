import { describe, expect, it } from 'vitest';
import {
	computeGridLevels,
	createLatentGrid,
	LATENT_INIT_SCALE,
	MAX_GRID_RESOLUTION
} from '../../../../examples/jsm/ntc/training/NTCGridModel.js';

describe( 'Addons > Neural > NeuralGridModel', () => {

	describe( 'computeGridLevels', () => {

		it( 'starts at baseResolution and multiplies by growthFactor each level', () => {

			expect( computeGridLevels( 16, 2, 4 ) ).toEqual( [ 16, 32, 64, 128 ] );

		} );

		it( 'returns a single resolution equal to baseResolution when levels <= 1, ignoring growthFactor', () => {

			expect( computeGridLevels( 16, 2, 1 ) ).toEqual( [ 16 ] );
			expect( computeGridLevels( 16, 3, 1 ) ).toEqual( [ 16 ] );

		} );

		it( 'clamps levels <= 0 to a single baseResolution level', () => {

			expect( computeGridLevels( 16, 2, 0 ) ).toEqual( [ 16 ] );
			expect( computeGridLevels( 16, 2, - 3 ) ).toEqual( [ 16 ] );

		} );

		it( 'produces a flat sequence when growthFactor is 1', () => {

			expect( computeGridLevels( 16, 1, 4 ) ).toEqual( [ 16, 16, 16, 16 ] );

		} );

		it( 'is monotonically non-decreasing across levels for any growthFactor >= 1', () => {

			const levels = computeGridLevels( 8, 1.5, 6 );

			for ( let i = 1; i < levels.length; i ++ ) {

				expect( levels[ i ] ).toBeGreaterThanOrEqual( levels[ i - 1 ] );

			}

		} );

		it( 'produces every resolution as a positive integer', () => {

			const levels = computeGridLevels( 4, 2.3, 7 );

			for ( const resolution of levels ) {

				expect( Number.isInteger( resolution ) ).toBe( true );
				expect( resolution ).toBeGreaterThanOrEqual( 1 );

			}

		} );

		it( 'every combination of baseResolution/growthFactor/levels is valid - no base > target degenerate case', () => {

			// Under the old (baseResolution, targetResolution, levels) API this
			// kind of input (a "target" smaller than the base) had to be special-
			// cased. Under the growthFactor API there's no such concept - any
			// growthFactor >= 1 just produces a non-decreasing sequence.
			const levels = computeGridLevels( 200, 2, 4 );

			expect( levels ).toEqual( [ 200, 400, 800, 1600 ] );

		} );

		it( 'treats baseResolution <= 0 as 1 rather than producing 0/NaN', () => {

			const levels = computeGridLevels( 0, 2, 3 );

			expect( levels.every( Number.isFinite ) ).toBe( true );
			expect( levels.every( ( r ) => r >= 1 ) ).toBe( true );
			expect( levels[ 0 ] ).toBe( 1 );

		} );

		it( 'rounds a non-integer baseResolution', () => {

			expect( computeGridLevels( 16.6, 2, 1 ) ).toEqual( [ 17 ] );

		} );

		it( 'omits levels whose resolution would exceed MAX_GRID_RESOLUTION rather than clamping them down', () => {

			// 64, 320, 1600, 8000, ... - the 4th level (8000) exceeds the cap, so
			// it (and every level after it, since resolution only grows) is
			// dropped instead of being clamped down to MAX_GRID_RESOLUTION.
			const levels = computeGridLevels( 64, 5, 8 );

			expect( levels ).toEqual( [ 64, 320, 1600 ] );
			expect( levels.every( ( r ) => r <= MAX_GRID_RESOLUTION ) ).toBe( true );

		} );

		it( 'returns an empty array when baseResolution itself already exceeds MAX_GRID_RESOLUTION', () => {

			expect( computeGridLevels( MAX_GRID_RESOLUTION + 1, 2, 4 ) ).toEqual( [] );

		} );

		it( 'returns exactly [ MAX_GRID_RESOLUTION ] when baseResolution is exactly at the cap', () => {

			expect( computeGridLevels( MAX_GRID_RESOLUTION, 2, 4 ) ).toEqual( [ MAX_GRID_RESOLUTION ] );

		} );

	} );

	describe( 'createLatentGrid', () => {

		it( 'sizes the backing Float32Array to width * height * channels', () => {

			const grid = createLatentGrid( 4, 3, 8, () => 0.5 );

			expect( grid.data.length ).toBe( 4 * 3 * 8 );
			expect( grid.data ).toBeInstanceOf( Float32Array );
			expect( grid.width ).toBe( 4 );
			expect( grid.height ).toBe( 3 );
			expect( grid.channels ).toBe( 8 );

		} );

		it( 'initializes every value within [-LATENT_INIT_SCALE, LATENT_INIT_SCALE]', () => {

			// random() spans its full [0,1) range across calls to hit both extremes
			let toggle = 0;
			const random = () => ( toggle ++ % 2 === 0 ? 0 : 1 - 1e-9 );
			const grid = createLatentGrid( 2, 2, 2, random );

			for ( const value of grid.data ) {

				expect( Math.abs( value ) ).toBeLessThanOrEqual( LATENT_INIT_SCALE + 1e-9 );

			}

		} );

		it( 'maps random() = 0.5 to exactly zero (the midpoint of the [-scale, scale] range)', () => {

			const grid = createLatentGrid( 1, 1, 4, () => 0.5 );

			expect( Array.from( grid.data ) ).toEqual( [ 0, 0, 0, 0 ] );

		} );

		it( 'maps random() = 0 and random() = 1 to the negative/positive extremes', () => {

			const gridMin = createLatentGrid( 1, 1, 1, () => 0 );
			const gridMax = createLatentGrid( 1, 1, 1, () => 1 );

			// Float32Array-backed, so only float32 precision (~1e-7), not float64.
			expect( gridMin.data[ 0 ] ).toBeCloseTo( - LATENT_INIT_SCALE, 5 );
			expect( gridMax.data[ 0 ] ).toBeCloseTo( LATENT_INIT_SCALE, 5 );

		} );

		it( 'produces a zero-length grid for zero channels without throwing', () => {

			const grid = createLatentGrid( 4, 4, 0, () => 0.5 );

			expect( grid.data.length ).toBe( 0 );

		} );

		it( 'calls random() once per data element (deterministic given a seeded sequence)', () => {

			let calls = 0;
			const random = () => {

				calls += 1;
				return 0.5;

			};

			createLatentGrid( 3, 2, 5, random );

			expect( calls ).toBe( 3 * 2 * 5 );

		} );

	} );

} );

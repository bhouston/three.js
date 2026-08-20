import { describe, expect, it } from 'vitest';
import { computeGridLevels, createLatentGrid, LATENT_INIT_SCALE } from '../../../../examples/jsm/neural/NeuralGridModel.js';

describe( 'Addons > Neural > NeuralGridModel', () => {

	describe( 'computeGridLevels', () => {

		it( 'returns a single resolution when levels <= 1, ignoring baseResolution', () => {

			expect( computeGridLevels( 16, 512, 1 ) ).toEqual( [ 512 ] );
			expect( computeGridLevels( 16, 512, 0 ) ).toEqual( [ 512 ] );

		} );

		it( 'rounds a sub-1 target down to the minimum resolution of 1', () => {

			expect( computeGridLevels( 16, 0.2, 1 ) ).toEqual( [ 1 ] );

		} );

		it( 'starts at baseResolution and ends at targetResolution', () => {

			const levels = computeGridLevels( 16, 256, 5 );

			expect( levels[ 0 ] ).toBe( 16 );
			expect( levels[ levels.length - 1 ] ).toBe( 256 );
			expect( levels.length ).toBe( 5 );

		} );

		it( 'is monotonically non-decreasing across levels (geometric spacing)', () => {

			const levels = computeGridLevels( 8, 300, 6 );

			for ( let i = 1; i < levels.length; i ++ ) {

				expect( levels[ i ] ).toBeGreaterThanOrEqual( levels[ i - 1 ] );

			}

		} );

		it( 'produces every resolution as a positive integer', () => {

			const levels = computeGridLevels( 4, 100, 7 );

			for ( const resolution of levels ) {

				expect( Number.isInteger( resolution ) ).toBe( true );
				expect( resolution ).toBeGreaterThanOrEqual( 1 );

			}

		} );

		it( 'handles baseResolution > targetResolution without producing a shrinking or invalid sequence', () => {

			// Degenerate/misconfigured input: still expected to clamp growth to
			// >= 1x rather than growing backwards or emitting resolutions below
			// baseResolution.
			const levels = computeGridLevels( 200, 16, 4 );

			expect( levels[ 0 ] ).toBe( 200 );
			for ( const resolution of levels ) {

				expect( resolution ).toBeGreaterThanOrEqual( 200 );

			}

		} );

		it( 'treats baseResolution <= 0 as 1 rather than dividing by zero', () => {

			const levels = computeGridLevels( 0, 64, 3 );

			expect( levels.every( Number.isFinite ) ).toBe( true );
			expect( levels.every( ( r ) => r >= 1 ) ).toBe( true );

		} );

		it( 'is stable when base and target resolution are equal (flat, not exponential, sequence)', () => {

			const levels = computeGridLevels( 32, 32, 4 );

			expect( levels ).toEqual( [ 32, 32, 32, 32 ] );

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

import { describe, expect, it } from 'vitest';
import {
	computeGridLevels,
	createLatentGrid,
	LATENT_INIT_SCALE,
	MAX_GRID_RESOLUTION,
	DEFAULT_MIPS_PER_LEVEL
} from '../../../../examples/jsm/ntc/training/NTCGridModel.js';

describe( 'Addons > NTC > NTCGridModel', () => {

	describe( 'computeGridLevels', () => {

		it( 'starts at baseResolution (finest, mip 0) and halves mipsPerLevel times per level', () => {

			expect( computeGridLevels( 128, 4, 2 ) ).toEqual( [ 128, 32, 8, 2 ] );

		} );

		it( 'defaults mipsPerLevel to DEFAULT_MIPS_PER_LEVEL when omitted', () => {

			expect( computeGridLevels( 128, 4 ) ).toEqual( computeGridLevels( 128, 4, DEFAULT_MIPS_PER_LEVEL ) );
			expect( DEFAULT_MIPS_PER_LEVEL ).toBe( 2 );

		} );

		it( 'mipsPerLevel = 1 matches a plain per-mip halving chain', () => {

			expect( computeGridLevels( 128, 4, 1 ) ).toEqual( [ 128, 64, 32, 16 ] );

		} );

		it( 'returns a single resolution equal to baseResolution when levels <= 1', () => {

			expect( computeGridLevels( 16, 1, 2 ) ).toEqual( [ 16 ] );

		} );

		it( 'clamps levels <= 0 to a single baseResolution level', () => {

			expect( computeGridLevels( 16, 0, 2 ) ).toEqual( [ 16 ] );
			expect( computeGridLevels( 16, - 3, 2 ) ).toEqual( [ 16 ] );

		} );

		it( 'never produces a resolution below 1, even for a level count deep enough to underflow', () => {

			const levels = computeGridLevels( 8, 6, 2 );

			expect( levels ).toEqual( [ 8, 2, 1, 1, 1, 1 ] );
			expect( levels.every( ( r ) => r >= 1 ) ).toBe( true );

		} );

		it( 'produces every resolution as a positive integer', () => {

			const levels = computeGridLevels( 100, 5, 2 );

			for ( const resolution of levels ) {

				expect( Number.isInteger( resolution ) ).toBe( true );
				expect( resolution ).toBeGreaterThanOrEqual( 1 );

			}

		} );

		it( 'treats baseResolution <= 0 as 1 rather than producing 0/NaN', () => {

			const levels = computeGridLevels( 0, 3, 2 );

			expect( levels.every( Number.isFinite ) ).toBe( true );
			expect( levels.every( ( r ) => r >= 1 ) ).toBe( true );
			expect( levels[ 0 ] ).toBe( 1 );

		} );

		it( 'rounds a non-integer baseResolution', () => {

			expect( computeGridLevels( 16.6, 1, 2 ) ).toEqual( [ 17 ] );

		} );

		it( 'clamps baseResolution to MAX_GRID_RESOLUTION rather than exceeding it', () => {

			const levels = computeGridLevels( MAX_GRID_RESOLUTION + 1000, 1, 2 );

			expect( levels ).toEqual( [ MAX_GRID_RESOLUTION ] );

		} );

		it( 'returns exactly [ MAX_GRID_RESOLUTION ] when baseResolution is exactly at the cap', () => {

			expect( computeGridLevels( MAX_GRID_RESOLUTION, 1, 2 ) ).toEqual( [ MAX_GRID_RESOLUTION ] );

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

import { describe, expect, it } from 'vitest';
import {
	FADE_BAND_MIPS,
	computeNaturalLod,
	computeAllNaturalLods,
	computeLevelLodWeight
} from '../../../../examples/jsm/ntc/NTCMipPyramid.js';

describe( 'Addons > NTC > NTCMipPyramid', () => {

	describe( 'computeNaturalLod', () => {

		it( 'is 0 when the grid resolution matches the texture resolution', () => {

			expect( computeNaturalLod( 512, 512 ) ).toBe( 0 );

		} );

		it( 'is log2(textureResolution / gridResolution) when the grid is coarser', () => {

			expect( computeNaturalLod( 128, 1024 ) ).toBeCloseTo( 3, 10 );
			expect( computeNaturalLod( 64, 1024 ) ).toBeCloseTo( 4, 10 );

		} );

		it( 'clamps to 0 rather than going negative when the grid is finer than the texture', () => {

			expect( computeNaturalLod( 1024, 128 ) ).toBe( 0 );

		} );

	} );

	describe( 'computeAllNaturalLods', () => {

		it( 'maps computeNaturalLod over every resolution, preserving order', () => {

			const resolutions = [ 16, 32, 64, 128 ];
			const textureResolution = 128;

			expect( computeAllNaturalLods( resolutions, textureResolution ) ).toEqual( [
				computeNaturalLod( 16, 128 ),
				computeNaturalLod( 32, 128 ),
				computeNaturalLod( 64, 128 ),
				computeNaturalLod( 128, 128 )
			] );

		} );

	} );

	describe( 'computeLevelLodWeight', () => {

		it( 'is 1 at or below the natural LOD', () => {

			expect( computeLevelLodWeight( 0, 3 ) ).toBe( 1 );
			expect( computeLevelLodWeight( 3, 3 ) ).toBe( 1 );

		} );

		it( 'fades linearly to 0 across exactly FADE_BAND_MIPS mips past the natural LOD', () => {

			const naturalLod = 2;

			expect( computeLevelLodWeight( naturalLod, naturalLod ) ).toBe( 1 );
			expect( computeLevelLodWeight( naturalLod + FADE_BAND_MIPS / 2, naturalLod ) ).toBeCloseTo( 0.5, 10 );
			expect( computeLevelLodWeight( naturalLod + FADE_BAND_MIPS, naturalLod ) ).toBe( 0 );

		} );

		it( 'clamps to 0 rather than going negative past the fade band', () => {

			expect( computeLevelLodWeight( 100, 0 ) ).toBe( 0 );

		} );

		it( 'clamps to 1 rather than exceeding it below the natural LOD', () => {

			expect( computeLevelLodWeight( - 5, 3 ) ).toBe( 1 );

		} );

	} );

} );

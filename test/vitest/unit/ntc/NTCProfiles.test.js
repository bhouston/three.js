import { describe, expect, it } from 'vitest';
import { NTC_PROFILES, NTC_PROFILE_NAMES, getNTCProfile } from '../../../../examples/jsm/ntc/training/NTCProfiles.js';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';
import { MLP_ACTIVATION_OPTIONS, GRID_LEVELS_OPTIONS, GRID_BASE_RESOLUTION_OPTIONS, MLP_HIDDEN_SIZE_OPTIONS } from '../../../../examples/jsm/ntc/training/NTCGridModel.js';

describe( 'Addons > NTC > NTCProfiles', () => {

	it( 'exposes exactly the three documented profile names', () => {

		expect( NTC_PROFILE_NAMES ).toEqual( [ 'mobile-fast', 'mobile-balanced', 'desktop-quality' ] );
		expect( Object.keys( NTC_PROFILES ) ).toEqual( NTC_PROFILE_NAMES );

	} );

	it( 'every profile only uses grid/MLP shape values also offered by NTCGridModel.js\'s GUI option lists', () => {

		for ( const name of NTC_PROFILE_NAMES ) {

			const profile = NTC_PROFILES[ name ];

			expect( GRID_LEVELS_OPTIONS ).toContain( profile.levels );
			expect( GRID_BASE_RESOLUTION_OPTIONS ).toContain( profile.baseResolution );
			expect( MLP_ACTIVATION_OPTIONS ).toContain( profile.hiddenActivation );

			for ( const hiddenSize of profile.hiddenSizes ) {

				expect( MLP_HIDDEN_SIZE_OPTIONS ).toContain( hiddenSize );

			}

		}

	} );

	it( 'every profile has a non-empty label and description', () => {

		for ( const name of NTC_PROFILE_NAMES ) {

			const profile = NTC_PROFILES[ name ];
			expect( typeof profile.label ).toBe( 'string' );
			expect( profile.label.length ).toBeGreaterThan( 0 );
			expect( typeof profile.description ).toBe( 'string' );
			expect( profile.description.length ).toBeGreaterThan( 0 );

		}

	} );

	it( 'mobile-fast is the cheapest profile: fewest hidden layers, narrowest hidden width, relu', () => {

		const mobileFast = NTC_PROFILES[ 'mobile-fast' ];

		expect( mobileFast.hiddenSizes.length ).toBe( 1 );
		expect( mobileFast.hiddenActivation ).toBe( 'relu' );

		for ( const other of [ 'mobile-balanced', 'desktop-quality' ] ) {

			expect( mobileFast.hiddenSizes[ 0 ] ).toBeLessThanOrEqual( NTC_PROFILES[ other ].hiddenSizes[ 0 ] );

		}

	} );

	it( 'desktop-quality is the only profile using hgelu, and has the widest hidden layers', () => {

		expect( NTC_PROFILES[ 'desktop-quality' ].hiddenActivation ).toBe( 'hgelu' );
		expect( NTC_PROFILES[ 'mobile-fast' ].hiddenActivation ).toBe( 'relu' );
		expect( NTC_PROFILES[ 'mobile-balanced' ].hiddenActivation ).toBe( 'relu' );

		const desktopWidth = NTC_PROFILES[ 'desktop-quality' ].hiddenSizes[ 0 ];
		expect( desktopWidth ).toBeGreaterThanOrEqual( NTC_PROFILES[ 'mobile-fast' ].hiddenSizes[ 0 ] );
		expect( desktopWidth ).toBeGreaterThanOrEqual( NTC_PROFILES[ 'mobile-balanced' ].hiddenSizes[ 0 ] );

	} );

	describe( 'getNTCProfile', () => {

		it( 'returns only the shape options (levels, baseResolution, hiddenSizes, hiddenActivation), not label/description', () => {

			const options = getNTCProfile( 'mobile-balanced' );

			expect( options ).toEqual( {
				levels: NTC_PROFILES[ 'mobile-balanced' ].levels,
				baseResolution: NTC_PROFILES[ 'mobile-balanced' ].baseResolution,
				hiddenSizes: NTC_PROFILES[ 'mobile-balanced' ].hiddenSizes,
				hiddenActivation: NTC_PROFILES[ 'mobile-balanced' ].hiddenActivation
			} );
			expect( options.label ).toBeUndefined();
			expect( options.description ).toBeUndefined();

		} );

		it( 'returns a copy of hiddenSizes, not a reference to the profile\'s own array', () => {

			const options = getNTCProfile( 'desktop-quality' );
			options.hiddenSizes.push( 999 );

			expect( NTC_PROFILES[ 'desktop-quality' ].hiddenSizes ).not.toContain( 999 );

		} );

		it( 'returns null for an unrecognized profile name', () => {

			expect( getNTCProfile( 'not-a-real-profile' ) ).toBeNull();

		} );

		it( 'can be spread directly into createNTCGridPyramidModel options and produces the expected shape', () => {

			for ( const name of NTC_PROFILE_NAMES ) {

				const profile = NTC_PROFILES[ name ];
				const options = { channels: 4, outputChannels: 3, ...getNTCProfile( name ) };
				const model = createNTCGridPyramidModel( options, () => 0.5 );

				expect( model.hiddenSizes ).toEqual( profile.hiddenSizes );
				expect( model.hiddenActivation ).toBe( profile.hiddenActivation );
				expect( model.resolutions[ 0 ] ).toBe( profile.baseResolution );
				expect( model.decoder.layers.length ).toBe( profile.hiddenSizes.length + 1 );

				for ( let i = 0; i < profile.hiddenSizes.length; i ++ ) {

					expect( model.decoder.layers[ i ].activation ).toBe( profile.hiddenActivation );

				}

			}

		} );

	} );

} );

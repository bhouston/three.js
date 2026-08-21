import { describe, expect, it } from 'vitest';
import { buildChannelActivations, CHANNELS, decodeConstantValues, getChannel, layoutChannels, MAX_TOTAL_CHANNELS } from '../../../../examples/jsm/neural-material/NeuralMaterialFormat.js';

describe( 'Addons > Neural > NeuralMaterial > NeuralMaterialFormat', () => {

	describe( 'getChannel', () => {

		it( 'returns the channel object for a known key', () => {

			const channel = getChannel( 'normal' );

			expect( channel.key ).toBe( 'normal' );
			expect( channel.size ).toBe( 2 );

		} );

		it( 'throws for an unknown key', () => {

			expect( () => getChannel( 'nonexistent' ) ).toThrow();

		} );

	} );

	describe( 'layoutChannels', () => {

		it( 'assigns contiguous flat offsets in the order the subset is given, not CHANNELS order', () => {

			const subset = [ getChannel( 'normal' ), getChannel( 'albedo' ) ];
			const { channels } = layoutChannels( subset );

			expect( channels[ 0 ].key ).toBe( 'normal' );
			expect( channels[ 0 ].offset ).toBe( 0 );
			expect( channels[ 1 ].key ).toBe( 'albedo' );
			expect( channels[ 1 ].offset ).toBe( 2 );

		} );

		it( 'sums channel sizes into totalChannels', () => {

			const subset = [ getChannel( 'normal' ), getChannel( 'albedo' ), getChannel( 'roughness' ) ];
			const { totalChannels } = layoutChannels( subset );

			expect( totalChannels ).toBe( 2 + 3 + 1 );

		} );

		it( 'computes packCount as ceil(totalChannels / 4), exactly on a 4-boundary', () => {

			// albedo(3) + roughness(1) = 4 -> exactly one pack, no off-by-one.
			const subset = [ getChannel( 'albedo' ), getChannel( 'roughness' ) ];
			const { totalChannels, packCount } = layoutChannels( subset );

			expect( totalChannels ).toBe( 4 );
			expect( packCount ).toBe( 1 );

		} );

		it( 'computes packCount as ceil(totalChannels / 4), one past a 4-boundary', () => {

			// albedo(3) + roughness(1) + metalness(1) = 5 -> spills into a second pack.
			const subset = [ getChannel( 'albedo' ), getChannel( 'roughness' ), getChannel( 'metalness' ) ];
			const { totalChannels, packCount } = layoutChannels( subset );

			expect( totalChannels ).toBe( 5 );
			expect( packCount ).toBe( 2 );

		} );

		it( 'computes packCount as ceil(totalChannels / 4), one short of a 4-boundary', () => {

			// normal(2) + roughness(1) = 3 -> still fits in one pack.
			const subset = [ getChannel( 'normal' ), getChannel( 'roughness' ) ];
			const { totalChannels, packCount } = layoutChannels( subset );

			expect( totalChannels ).toBe( 3 );
			expect( packCount ).toBe( 1 );

		} );

		it( 'returns totalChannels 0, packCount 0 and an empty channels array for an empty subset', () => {

			const layout = layoutChannels( [] );

			expect( layout.totalChannels ).toBe( 0 );
			expect( layout.packCount ).toBe( Math.ceil( 0 / 4 ) );
			expect( layout.channels ).toEqual( [] );

		} );

	} );

	describe( 'MAX_TOTAL_CHANNELS', () => {

		it( 'equals the sum of every channel size in CHANNELS', () => {

			const expected = CHANNELS.reduce( ( sum, channel ) => sum + channel.size, 0 );

			expect( MAX_TOTAL_CHANNELS ).toBe( expected );

		} );

		it( 'equals layoutChannels(CHANNELS).totalChannels', () => {

			expect( MAX_TOTAL_CHANNELS ).toBe( layoutChannels( CHANNELS ).totalChannels );

		} );

	} );

	describe( 'buildChannelActivations', () => {

		it( 'emits one activation string per output component, at the flat index matching layoutChannels offsets', () => {

			const subset = [ getChannel( 'albedo' ), getChannel( 'normal' ), getChannel( 'roughness' ) ];
			const { channels } = layoutChannels( subset );
			const activations = buildChannelActivations( channels );

			// albedo: size 3, sigmoid -> indices 0,1,2
			// normal: size 2, tanh -> indices 3,4
			// roughness: size 1, sigmoid -> index 5
			expect( activations ).toEqual( [
				'sigmoid', 'sigmoid', 'sigmoid',
				'tanh', 'tanh',
				'sigmoid'
			] );

		} );

		it( 'returns an empty array for an empty channel list', () => {

			expect( buildChannelActivations( [] ) ).toEqual( [] );

		} );

	} );

	describe( 'iridescenceThickness range metadata (never trained, never adapted)', () => {

		it( 'iridescenceThicknessRangeMin/Max always classify as constant, never active, regardless of what nodes the material sets', () => {

			const material = {
				iridescenceThicknessNode: {},
				iridescenceThicknessRange: [ 50, 900 ]
			};

			const rangeMin = getChannel( 'iridescenceThicknessRangeMin' );
			const rangeMax = getChannel( 'iridescenceThicknessRangeMax' );

			// No nodeKeys check can ever make these "active" - see
			// NeuralMaterialFormat.js's `iridescenceThicknessRangeChannel` doc
			// comment - so they always resolve straight off the source
			// material's own declared range, verbatim.
			expect( rangeMin.resolveConstant( material ) ).toBe( 50 );
			expect( rangeMax.resolveConstant( material ) ).toBe( 900 );

		} );

		it( 'fall back to three.js\'s own default range ([100, 400]) when the material never set one', () => {

			const rangeMin = getChannel( 'iridescenceThicknessRangeMin' );
			const rangeMax = getChannel( 'iridescenceThicknessRangeMax' );

			expect( rangeMin.resolveConstant( {} ) ).toBe( 100 );
			expect( rangeMax.resolveConstant( {} ) ).toBe( 400 );

		} );

		it( 'applyConstant writes straight into targetMaterial.iridescenceThicknessRange at its own index, no decoding', () => {

			const rangeMin = getChannel( 'iridescenceThicknessRangeMin' );
			const rangeMax = getChannel( 'iridescenceThicknessRangeMax' );
			const target = { iridescenceThicknessRange: [ 100, 400 ] };

			rangeMin.applyConstant( target, 50 );
			rangeMax.applyConstant( target, 900 );

			expect( target.iridescenceThicknessRange ).toEqual( [ 50, 900 ] );

		} );

		it( 'iridescenceThickness itself resolves as a [0,1] fraction of the material\'s own range, not a raw nanometer value', () => {

			const channel = getChannel( 'iridescenceThickness' );

			// No map at all -> always fraction 1 (the declared maximum),
			// matching MaterialNode.js's own IRIDESCENCE_THICKNESS fallback -
			// never a raw nanometer default.
			expect( channel.resolveConstant( {} ) ).toBe( 1 );
			expect( channel.defaultValue ).toBe( 1 );
			expect( channel.clampRange ).toEqual( [ 0, 1 ] );
			expect( channel.activation ).toBe( 'sigmoid' );

		} );

		it( 'iridescenceThickness.applyConstant decodes the fraction back to nanometers using targetMaterial\'s own (already-applied) range', () => {

			const channel = getChannel( 'iridescenceThickness' );
			// Simulates the range channels above having already run first in
			// the same apply loop (see NeuralMaterialNodeMaterial's
			// constructor) - exactly the ordering `CHANNELS` itself relies on.
			const target = { iridescenceThicknessRange: [ 50, 900 ] };

			channel.applyConstant( target, 0.5 ); // fraction 0.5 -> halfway between 50 and 900
			expect( target.iridescenceThicknessNode.node.value ).toBeCloseTo( 50 + 0.5 * ( 900 - 50 ), 5 );

		} );

	} );

	describe( 'fixedRangeScalarChannel (iridescenceIOR, ior)', () => {

		it( 'trains strictly within a fixed [min, max] window, not an unbounded value merely clamped after the fact', () => {

			for ( const key of [ 'iridescenceIOR', 'ior' ] ) {

				const channel = getChannel( key );

				expect( channel.activation ).toBe( 'sigmoid' );
				// The physical bound lives in the closure the trained slice is
				// decoded through (applyActive), not in a separate post-hoc
				// `clampRange` the way `simpleScalarChannel`-built channels use.
				expect( channel.clampRange ).toBeNull();

			}

		} );

		it( 'resolveConstant/applyConstant round-trip the real physical value (not a fraction) on the constant path', () => {

			const channel = getChannel( 'iridescenceIOR' );

			expect( channel.resolveConstant( {} ) ).toBe( channel.defaultValue ); // 1.3, unset
			expect( channel.resolveConstant( { iridescenceIOR: 1.8 } ) ).toBe( 1.8 );

			const target = {};
			channel.applyConstant( target, 1.8 );
			expect( target.iridescenceIORNode.node.value ).toBeCloseTo( 1.8, 5 );

		} );

	} );

	describe( 'attenuationDistance', () => {

		it( 'encodes an unset (Infinity) attenuationDistance as the finite sentinel, matching its own defaultValue', () => {

			const channel = getChannel( 'attenuationDistance' );

			expect( channel.resolveConstant( {} ) ).toBe( channel.defaultValue );
			expect( Number.isFinite( channel.defaultValue ) ).toBe( true );

		} );

		it( 'encodes an explicit finite attenuationDistance as-is', () => {

			const channel = getChannel( 'attenuationDistance' );

			expect( channel.resolveConstant( { attenuationDistance: 2.5 } ) ).toBe( 2.5 );

		} );

		it( 'JSON round-trips the sentinel without loss (unlike Infinity itself)', () => {

			const channel = getChannel( 'attenuationDistance' );
			const encoded = channel.resolveConstant( {} );

			expect( JSON.parse( JSON.stringify( { value: encoded } ) ).value ).toBe( encoded );
			expect( JSON.parse( JSON.stringify( { value: Infinity } ) ).value ).toBeNull(); // what we're avoiding

		} );

	} );

	describe( 'decodeConstantValues', () => {

		it( 'decodes the attenuationDistance sentinel back into Infinity', () => {

			const channel = getChannel( 'attenuationDistance' );
			const decoded = decodeConstantValues( { attenuationDistance: channel.defaultValue } );

			expect( decoded.attenuationDistance ).toBe( Infinity );

		} );

		it( 'leaves a non-sentinel attenuationDistance value untouched', () => {

			const decoded = decodeConstantValues( { attenuationDistance: 2.5 } );

			expect( decoded.attenuationDistance ).toBe( 2.5 );

		} );

		it( 'leaves channels without a decodeConstant hook untouched', () => {

			const decoded = decodeConstantValues( { roughness: 0.4 } );

			expect( decoded.roughness ).toBe( 0.4 );

		} );

		it( 'passes unknown keys through unchanged', () => {

			const decoded = decodeConstantValues( { notAChannel: 42 } );

			expect( decoded.notAChannel ).toBe( 42 );

		} );

	} );

} );

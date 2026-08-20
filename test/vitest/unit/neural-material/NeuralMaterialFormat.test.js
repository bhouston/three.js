import { describe, expect, it } from 'vitest';
import { buildChannelActivations, CHANNELS, getChannel, layoutChannels, MAX_TOTAL_CHANNELS } from '../../../../examples/jsm/neural-material/NeuralMaterialFormat.js';

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

} );

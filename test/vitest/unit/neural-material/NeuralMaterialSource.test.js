import { describe, expect, it } from 'vitest';
import { CHANNELS, layoutChannels } from '../../../../examples/jsm/neural-material/NeuralMaterialFormat.js';
import { classifyMaterialChannels } from '../../../../examples/jsm/neural-material/NeuralMaterialSource.js';

describe( 'Addons > Neural > NeuralMaterial > NeuralMaterialSource', () => {

	describe( 'classifyMaterialChannels', () => {

		it( 'classifies a channel active iff at least one of its nodeKeys is truthy on the material', () => {

			const material = { colorNode: {}, roughnessNode: {} };
			const { activeChannels } = classifyMaterialChannels( material );

			const activeKeys = activeChannels.map( ( channel ) => channel.key ).sort();

			expect( activeKeys ).toEqual( [ 'albedo', 'roughness' ].sort() );

		} );

		it( 'classifies a channel constant when none of its nodeKeys are set, resolving its constant value from a plain property', () => {

			const material = { roughness: 0.4 };
			const { activeChannels, constantValues } = classifyMaterialChannels( material );

			expect( activeChannels.some( ( channel ) => channel.key === 'roughness' ) ).toBe( false );
			expect( constantValues.roughness ).toBe( 0.4 );

		} );

		it( 'falls back to the documented hardcoded defaults for a completely empty material', () => {

			const { activeChannels, constantValues } = classifyMaterialChannels( {} );

			expect( activeChannels.length ).toBe( 0 );

			expect( constantValues.albedo ).toEqual( [ 1, 1, 1 ] );
			expect( constantValues.opacity ).toBe( 1 );
			expect( constantValues.normal ).toEqual( [ 0, 0, 1 ] );
			expect( constantValues.roughness ).toBe( 1 );
			expect( constantValues.metalness ).toBe( 0 );
			expect( constantValues.clearcoat ).toBe( 0 );
			expect( constantValues.clearcoatRoughness ).toBe( 0 );
			expect( constantValues.clearcoatNormal ).toEqual( [ 0, 0, 1 ] );
			expect( constantValues.transmission ).toBe( 0 );
			expect( constantValues.emissive ).toEqual( [ 0, 0, 0 ] );
			expect( constantValues.anisotropy ).toEqual( [ 0, 0 ] );
			expect( constantValues.sheenColor ).toEqual( [ 0, 0, 0 ] );
			expect( constantValues.sheenRoughness ).toBe( 1 );

		} );

		it( 'produces totalChannels/packCount consistent with layoutChannels applied to just the active subset', () => {

			const material = { colorNode: {}, normalNode: {}, roughnessNode: {}, metalnessNode: {} };
			const result = classifyMaterialChannels( material );

			const expectedActiveKeys = CHANNELS.filter( ( channel ) =>
				channel.nodeKeys.some( ( key ) => Boolean( material[ key ] ) )
			);
			const expected = layoutChannels( expectedActiveKeys );

			expect( result.totalChannels ).toBe( expected.totalChannels );
			expect( result.packCount ).toBe( expected.packCount );
			expect( result.activeChannels.map( ( channel ) => channel.key ) ).toEqual(
				expected.channels.map( ( channel ) => channel.key )
			);

		} );

		it( 'with every possible nodeKey set, every channel is classified active', () => {

			const material = {};
			for ( const channel of CHANNELS ) {

				for ( const key of channel.nodeKeys ) material[ key ] = {};

			}

			expect( () => classifyMaterialChannels( material ) ).not.toThrow();

			const { activeChannels, constantValues } = classifyMaterialChannels( material );

			expect( activeChannels.length ).toBe( CHANNELS.length );
			expect( Object.keys( constantValues ).length ).toBe( 0 );

		} );

		it( 'with no nodeKeys set, every channel is classified constant', () => {

			expect( () => classifyMaterialChannels( {} ) ).not.toThrow();

			const { activeChannels, constantValues } = classifyMaterialChannels( {} );

			expect( activeChannels.length ).toBe( 0 );
			expect( Object.keys( constantValues ).length ).toBe( CHANNELS.length );

		} );

	} );

} );

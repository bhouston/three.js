import { describe, expect, it } from 'vitest';
import { createNTCGridPyramidModel } from '../../../../examples/jsm/ntc/training/NTCGridPyramidModel.js';
import { CHANNELS, layoutChannels, getChannel } from '../../../../examples/jsm/ntc/NTCFormat.js';
import { encodeNTC, FORMAT, VERSION } from '../../../../examples/jsm/ntc/training/NTCManifest.js';
import { NTCLoader } from '../../../../examples/jsm/loaders/NTCLoader.js';

// Mirrors what `NTCSource.classifyMaterialChannels` would return
// for a material whose 'albedo' and 'roughness' channels are node-driven
// (active/trained) and everything else is a flat constant - without needing
// a real THREE.Material/renderer to drive that classification.
function buildChannelClassification() {

	const activeKeys = [ 'albedo', 'roughness' ];
	const activeList = CHANNELS.filter( ( channel ) => activeKeys.includes( channel.key ) );
	const { channels: activeChannels, totalChannels, packCount } = layoutChannels( activeList );

	const constantValues = {};
	for ( const channel of CHANNELS ) {

		if ( activeKeys.includes( channel.key ) === false ) constantValues[ channel.key ] = channel.defaultValue;

	}

	return { activeChannels, totalChannels, packCount, constantValues };

}

function buildModel( outputChannels ) {

	const random = ( () => {

		let seed = 54321;
		return () => {

			seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
			return seed / 0x7fffffff;

		};

	} )();

	return createNTCGridPyramidModel( {
		channels: 4,
		levels: 2,
		baseResolution: 4,
		growthFactor: 2,
		hiddenSizes: [ 6 ],
		outputChannels
	}, random );

}

describe( 'Addons > NTC > NTCManifest', () => {

	it( 'FORMAT/VERSION are the expected identifiers', () => {

		expect( FORMAT ).toBe( 'three-ntc' );
		expect( VERSION ).toBe( 1 );

	} );

	it( 'round-trips a channel-packed model + channel classification through export -> JSON -> load', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );

		const manifest = encodeNTC( model, classification, { name: 'roundtrip material' } );

		expect( manifest.format ).toBe( FORMAT );
		expect( manifest.version ).toBe( VERSION );
		expect( manifest.latents.levels.length ).toBe( model.grids.length );
		expect( manifest.channels.activeKeys ).toEqual( [ 'albedo', 'roughness' ] );

		const json = JSON.parse( JSON.stringify( manifest ) );
		const loaded = new NTCLoader().parse( json );

		expect( loaded.name ).toBe( 'roundtrip material' );

		// cpuModel reconstructs the same shape createNTCGridPyramidModel
		// produces - here the focus is the channel-classification wrapper.
		expect( loaded.cpuModel.outputChannels ).toBe( classification.totalChannels );
		expect( loaded.cpuModel.decoder.layers[ loaded.cpuModel.decoder.layers.length - 1 ].outputSize ).toBe( classification.totalChannels );

		// channelClassification: the shape NTCNodeMaterial's
		// constructor destructures - { activeChannels, constantValues } - and
		// sliceChannels indexes via each entry's key/size/offset/activation.
		expect( loaded.channelClassification.totalChannels ).toBe( classification.totalChannels );
		expect( loaded.channelClassification.packCount ).toBe( classification.packCount );
		expect( loaded.channelClassification.activeChannels.map( ( c ) => c.key ) ).toEqual( [ 'albedo', 'roughness' ] );

		for ( let i = 0; i < classification.activeChannels.length; i ++ ) {

			const original = classification.activeChannels[ i ];
			const loadedChannel = loaded.channelClassification.activeChannels[ i ];

			expect( loadedChannel.size ).toBe( original.size );
			expect( loadedChannel.activation ).toBe( original.activation );
			expect( loadedChannel.clampRange ).toEqual( original.clampRange );
			expect( loadedChannel.offset ).toBe( original.offset );

		}

		expect( loaded.channelClassification.constantValues ).toEqual( classification.constantValues );

	} );

	it( 'getChannel resolves every stored activeKey back to the exact same CHANNELS entry (no drift between manifest and format table)', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );

		const manifest = encodeNTC( model, classification, { name: 'consistency' } );

		for ( const key of manifest.channels.activeKeys ) {

			expect( () => getChannel( key ) ).not.toThrow();

		}

	} );

	it( 'encodeNTC produces the flat .ntc manifest shape (no nested texture block)', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );

		const exported = encodeNTC( model, classification, { name: 'exported' } );

		expect( exported.format ).toBe( FORMAT );
		expect( exported.name ).toBe( 'exported' );
		expect( exported.latents ).toBeDefined();
		expect( exported.mlp ).toBeDefined();
		expect( exported.texture ).toBeUndefined();

	} );

	it( 'round-trips renderFlags (side/transparent) through export -> JSON -> load', () => {

		const classification = { ...buildChannelClassification(), renderFlags: { side: 2 /* THREE.DoubleSide */, transparent: true } };
		const model = buildModel( classification.totalChannels );

		const manifest = encodeNTC( model, classification, { name: 'renderFlags roundtrip' } );
		expect( manifest.renderFlags ).toEqual( { side: 2, transparent: true } );

		const json = JSON.parse( JSON.stringify( manifest ) );
		const loaded = new NTCLoader().parse( json );

		expect( loaded.channelClassification.renderFlags ).toEqual( { side: 2, transparent: true } );

	} );

	it( 'a manifest saved without renderFlags (pre-existing files) loads with renderFlags null', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );

		const manifest = encodeNTC( model, classification, { name: 'no renderFlags' } );
		expect( manifest.renderFlags ).toBeNull();

		const loaded = new NTCLoader().parse( JSON.parse( JSON.stringify( manifest ) ) );

		expect( loaded.channelClassification.renderFlags ).toBeNull();

	} );

	it( 'loader rejects a manifest with the wrong format', () => {

		const loader = new NTCLoader();

		expect( () => loader.parse( { format: 'not-an-ntc-manifest', version: 1 } ) ).toThrow( /Unsupported format/ );

	} );

	it( 'loader rejects a manifest with an unknown channel key', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );
		const manifest = encodeNTC( model, classification, { name: 'bad channel' } );
		manifest.channels.activeKeys.push( 'notARealChannel' );

		const loader = new NTCLoader();

		expect( () => loader.parse( manifest ) ).toThrow( /unknown channel/ );

	} );

} );

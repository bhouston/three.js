import { describe, expect, it } from 'vitest';
import { createNeuralTextureModel } from '../../../../examples/jsm/neural-texture/NeuralTextureModel.js';
import { CHANNELS, layoutChannels, getChannel } from '../../../../examples/jsm/neural-material/NeuralMaterialFormat.js';
import { createNeuralMaterialManifest, exportNeuralMaterial, FORMAT, VERSION } from '../../../../examples/jsm/neural-material/NeuralMaterialManifest.js';
import { NeuralMaterialLoader } from '../../../../examples/jsm/loaders/NeuralMaterialLoader.js';

// Mirrors what `NeuralMaterialSource.classifyMaterialChannels` would return
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

	return createNeuralTextureModel( {
		channels: 4,
		levels: 2,
		baseResolution: 4,
		growthFactor: 2,
		hiddenSizes: [ 6 ],
		outputChannels
	}, random );

}

describe( 'Addons > Neural > Neural-Material > NeuralMaterialManifest', () => {

	it( 'FORMAT/VERSION are the expected identifiers', () => {

		expect( FORMAT ).toBe( 'three-neural-material' );
		expect( VERSION ).toBe( 1 );

	} );

	it( 'round-trips a channel-packed model + channel classification through export -> JSON -> load', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );

		const manifest = createNeuralMaterialManifest( model, classification, { name: 'roundtrip material' } );

		expect( manifest.format ).toBe( FORMAT );
		expect( manifest.version ).toBe( VERSION );
		expect( manifest.texture.format ).toBe( 'three-neural-texture' );
		expect( manifest.channels.activeKeys ).toEqual( [ 'albedo', 'roughness' ] );

		const json = JSON.parse( JSON.stringify( manifest ) );
		const loaded = new NeuralMaterialLoader().parse( json );

		expect( loaded.name ).toBe( 'roundtrip material' );

		// cpuModel reconstructs the same shape NeuralTextureLoader produces -
		// see NeuralTextureManifest.test.js for the per-value tolerance checks
		// of that path; here the focus is the channel-classification wrapper.
		expect( loaded.cpuModel.outputChannels ).toBe( classification.totalChannels );
		expect( loaded.cpuModel.decoder.layers[ loaded.cpuModel.decoder.layers.length - 1 ].outputSize ).toBe( classification.totalChannels );

		// channelClassification: the shape NeuralMaterialNodeMaterial's
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

		const manifest = createNeuralMaterialManifest( model, classification, { name: 'consistency' } );

		for ( const key of manifest.channels.activeKeys ) {

			expect( () => getChannel( key ) ).not.toThrow();

		}

	} );

	it( 'exportNeuralMaterial returns the same shape as createNeuralMaterialManifest', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );

		const exported = exportNeuralMaterial( model, classification, { name: 'exported' } );

		expect( exported.format ).toBe( FORMAT );
		expect( exported.name ).toBe( 'exported' );
		expect( exported.texture ).toBeDefined();

	} );

	it( 'round-trips renderFlags (side/transparent) through export -> JSON -> load', () => {

		const classification = { ...buildChannelClassification(), renderFlags: { side: 2 /* THREE.DoubleSide */, transparent: true } };
		const model = buildModel( classification.totalChannels );

		const manifest = createNeuralMaterialManifest( model, classification, { name: 'renderFlags roundtrip' } );
		expect( manifest.renderFlags ).toEqual( { side: 2, transparent: true } );

		const json = JSON.parse( JSON.stringify( manifest ) );
		const loaded = new NeuralMaterialLoader().parse( json );

		expect( loaded.channelClassification.renderFlags ).toEqual( { side: 2, transparent: true } );

	} );

	it( 'a manifest saved without renderFlags (pre-existing files) loads with renderFlags null', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );

		const manifest = createNeuralMaterialManifest( model, classification, { name: 'no renderFlags' } );
		expect( manifest.renderFlags ).toBeNull();

		const loaded = new NeuralMaterialLoader().parse( JSON.parse( JSON.stringify( manifest ) ) );

		expect( loaded.channelClassification.renderFlags ).toBeNull();

	} );

	it( 'loader rejects a manifest with the wrong format', () => {

		const loader = new NeuralMaterialLoader();

		expect( () => loader.parse( { format: 'not-a-neural-material', version: 1 } ) ).toThrow( /Unsupported format/ );

	} );

	it( 'loader rejects a manifest with an unknown channel key', () => {

		const classification = buildChannelClassification();
		const model = buildModel( classification.totalChannels );
		const manifest = createNeuralMaterialManifest( model, classification, { name: 'bad channel' } );
		manifest.channels.activeKeys.push( 'notARealChannel' );

		const loader = new NeuralMaterialLoader();

		expect( () => loader.parse( manifest ) ).toThrow( /unknown channel/ );

	} );

} );

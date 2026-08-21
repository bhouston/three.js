import { describe, expect, it } from 'vitest';
import { createNeuralTextureModel } from '../../../../examples/jsm/neural-texture/NeuralTextureModel.js';
import { createNeuralTextureManifest, exportNeuralTexture, FORMAT, VERSION } from '../../../../examples/jsm/neural-texture/NeuralTextureManifest.js';
import { NeuralTextureLoader } from '../../../../examples/jsm/loaders/NeuralTextureLoader.js';

// Round-trips a real `createNeuralTextureModel` CPU model through
// `createNeuralTextureManifest` -> JSON.stringify/parse (to exercise the
// exact base64/JSON path a real save/load cycle uses, not just the plain JS
// object) -> `NeuralTextureLoader.parse`, then checks the loaded CPU model
// reconstructs the original within each encoding's own precision budget:
// one quantization step (`(max-min)/255`) for the uint8 latent grid, and
// float16 machine epsilon (~2^-10 relative) for the MLP weights/biases.
function buildModel() {

	const random = ( () => {

		let seed = 12345;
		return () => {

			seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
			return seed / 0x7fffffff;

		};

	} )();

	return createNeuralTextureModel( {
		channels: 4,
		levels: 3,
		baseResolution: 4,
		growthFactor: 2,
		hiddenSizes: [ 6, 6 ],
		outputChannels: 3,
		peOctaves: 2,
		inputEncoding: 'positional'
	}, random );

}

function quantizationTolerance( min, max ) {

	return ( max - min ) / 255 + 1e-6;

}

const FLOAT16_RELATIVE_TOLERANCE = 6e-3; // ~2^-10 plus a little slack for accumulated rounding

function float16Tolerance( value ) {

	return Math.max( Math.abs( value ) * FLOAT16_RELATIVE_TOLERANCE, 2e-3 );

}

describe( 'Addons > Neural > Neural-Texture > NeuralTextureManifest', () => {

	it( 'FORMAT/VERSION are the expected identifiers', () => {

		expect( FORMAT ).toBe( 'three-neural-texture' );
		expect( VERSION ).toBe( 1 );

	} );

	it( 'round-trips grid data (uint8) and MLP weights/biases (float16) through export -> JSON -> load within tolerance', () => {

		const model = buildModel();
		const manifest = createNeuralTextureManifest( model, { name: 'roundtrip test' } );

		expect( manifest.format ).toBe( FORMAT );
		expect( manifest.version ).toBe( VERSION );
		expect( manifest.latents.levels.length ).toBe( model.grids.length );

		// Exercise the real base64/JSON path, not just the plain object.
		const json = JSON.parse( JSON.stringify( manifest ) );

		const loader = new NeuralTextureLoader();
		const loaded = loader.parse( json );

		expect( loaded.channels ).toBe( model.channels );
		expect( loaded.levels ).toBe( model.grids.length );
		expect( loaded.grids.length ).toBe( model.grids.length );
		expect( loaded.peOctaves ).toBe( model.peOctaves );
		expect( loaded.inputEncoding ).toBe( model.inputEncoding );
		expect( loaded.outputChannels ).toBe( model.outputChannels );

		for ( let g = 0; g < model.grids.length; g ++ ) {

			const originalGrid = model.grids[ g ];
			const loadedGrid = loaded.grids[ g ];
			const level = json.latents.levels[ g ];
			const tolerance = quantizationTolerance( level.min, level.max );

			expect( loadedGrid.width ).toBe( originalGrid.width );
			expect( loadedGrid.height ).toBe( originalGrid.height );
			expect( loadedGrid.channels ).toBe( originalGrid.channels );
			expect( loadedGrid.data.length ).toBe( originalGrid.data.length );

			for ( let i = 0; i < originalGrid.data.length; i ++ ) {

				expect( Math.abs( loadedGrid.data[ i ] - originalGrid.data[ i ] ) ).toBeLessThanOrEqual( tolerance );

			}

		}

		expect( loaded.decoder.layers.length ).toBe( model.decoder.layers.length );

		for ( let l = 0; l < model.decoder.layers.length; l ++ ) {

			const originalLayer = model.decoder.layers[ l ];
			const loadedLayer = loaded.decoder.layers[ l ];

			expect( loadedLayer.inputSize ).toBe( originalLayer.inputSize );
			expect( loadedLayer.outputSize ).toBe( originalLayer.outputSize );
			expect( loadedLayer.activation ).toBe( originalLayer.activation );

			for ( let i = 0; i < originalLayer.weights.length; i ++ ) {

				expect( Math.abs( loadedLayer.weights[ i ] - originalLayer.weights[ i ] ) ).toBeLessThanOrEqual( float16Tolerance( originalLayer.weights[ i ] ) );

			}

			for ( let i = 0; i < originalLayer.biases.length; i ++ ) {

				expect( Math.abs( loadedLayer.biases[ i ] - originalLayer.biases[ i ] ) ).toBeLessThanOrEqual( float16Tolerance( originalLayer.biases[ i ] ) );

			}

		}

	} );

	it( 'uses cpuModel.quantizationRange when present instead of re-scanning', () => {

		const model = buildModel();
		model.quantizationRange = model.grids.map( () => [ -5, 5 ] );

		const manifest = createNeuralTextureManifest( model, { name: 'explicit range' } );

		for ( const level of manifest.latents.levels ) {

			expect( level.min ).toBe( -5 );
			expect( level.max ).toBe( 5 );

		}

	} );

	it( 'exportNeuralTexture returns the same shape as createNeuralTextureManifest', () => {

		const model = buildModel();

		const exported = exportNeuralTexture( model, { name: 'exported' } );

		expect( exported.format ).toBe( FORMAT );
		expect( exported.name ).toBe( 'exported' );
		expect( Array.isArray( exported.latents.levels ) ).toBe( true );

	} );

	it( 'produces a CPU model that carries the exact fields NeuralTextureNodeMaterial.js\'s buildLevelTextures/evaluateNeuralTextureRaw destructure', () => {

		// Verifies field-level integration without importing
		// NeuralTextureNodeMaterial.js itself, which extends THREE.NodeMaterial
		// and therefore requires the WebGPU three build ('browser' vitest
		// project's 'three' alias) just to import - see vitest.config.js.
		const model = buildModel();
		const manifest = createNeuralTextureManifest( model, { name: 'shape test' } );
		const json = JSON.parse( JSON.stringify( manifest ) );
		const loaded = new NeuralTextureLoader().parse( json );

		// buildLevelTextures: cpuModel.grids[i].{data,width,height,channels}
		for ( const grid of loaded.grids ) {

			expect( grid.data instanceof Float32Array ).toBe( true );
			expect( typeof grid.width ).toBe( 'number' );
			expect( typeof grid.height ).toBe( 'number' );
			expect( typeof grid.channels ).toBe( 'number' );

		}

		// evaluateNeuralTextureRaw: cpuModel.grids[i].channels, cpuModel.inputEncoding,
		// cpuModel.peOctaves, cpuModel.decoder.layers[i].{weights,biases,inputSize,outputSize,activation}
		expect( typeof loaded.inputEncoding ).toBe( 'string' );
		expect( typeof loaded.peOctaves ).toBe( 'number' );

		for ( const layer of loaded.decoder.layers ) {

			expect( layer.weights instanceof Float32Array ).toBe( true );
			expect( layer.biases instanceof Float32Array ).toBe( true );
			expect( layer.weights.length ).toBe( layer.inputSize * layer.outputSize );
			expect( layer.biases.length ).toBe( layer.outputSize );

		}

	} );

	it( 'loader rejects a manifest with the wrong format', () => {

		const loader = new NeuralTextureLoader();

		expect( () => loader.parse( { format: 'not-a-neural-texture', version: 1 } ) ).toThrow( /Unsupported format/ );

	} );

	it( 'loader rejects a manifest with the wrong version', () => {

		const model = buildModel();
		const manifest = createNeuralTextureManifest( model, { name: 'bad version' } );
		manifest.version = 999;

		const loader = new NeuralTextureLoader();

		expect( () => loader.parse( manifest ) ).toThrow( /Unsupported version/ );

	} );

} );

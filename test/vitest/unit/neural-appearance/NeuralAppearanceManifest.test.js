import { describe, expect, it } from 'vitest';
import {
	createNeuralAppearanceManifest,
	serializeLayers
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceManifest.js';
import { decodeUint8Base64 } from '../../../../examples/jsm/neural/NeuralBinaryCodec.js';

// `createNeuralAppearanceManifest` is the "encoder" half of the manifest
// format: model (plain JS objects - grids/MLPs, the same shapes `createModel`
// in NeuralAppearanceModel.js produces) -> plain JSON manifest. These tests
// build minimal hand-rolled model objects (mirroring how
// NeuralAppearanceModel.test.js's `makeGrid` helper works) rather than going
// through `createModel`, so each conditional/offset path can be exercised in
// isolation without pulling in grid-sampling geometry.
//
// Deliberately does NOT import NeuralAppearanceFormat.js's `computeLatent
// Channels`/`computeDecoderInputSize`/`computeIblInputSize`/
// `computeIndirectInputSize` to compute *expected* values below - that would
// just be asking the source's own formula whether it agrees with itself
// (passes by construction, catches nothing). The `expectedXxx` helpers
// re-derive the same arithmetic independently, from the architecture
// documented and separately verified in NeuralAppearanceModel.test.js: a
// latent vector is `levels` grids x `CHANNELS_PER_LEVEL` (a raw config
// constant, not a formula) each; the decoder input appends a 12-wide TBN
// frame pair (see forwardDecoderInput's `output.length === latents.length +
// 12` assertion); the IBL head appends 6 (two 3-wide dot-product triples,
// see forwardIBLInput's `IBL_INPUT_SIZE` assertion); the indirect-probe
// heads append `wo` (3) + probe (3) (see buildIndirectProbeInput).
const CHANNELS_PER_LEVEL = 4;

function expectedLatentChannels( levels ) {

	return levels * CHANNELS_PER_LEVEL;

}

function expectedDecoderInputSize( levels ) {

	return expectedLatentChannels( levels ) + 12;

}

function expectedIblInputSize( levels ) {

	return expectedLatentChannels( levels ) + 6;

}

function expectedIndirectInputSize( levels ) {

	return expectedLatentChannels( levels ) + 3 + 3;

}

function makeGrid( width, height, channels, fill = 0.25 ) {

	return { width, height, channels, data: new Array( width * height * channels ).fill( fill ) };

}

function makeMLP( inputSize, outputSize, activation = 'relu' ) {

	return {
		layers: [
			{
				inputSize,
				outputSize,
				activation,
				weights: new Array( inputSize * outputSize ).fill( 0 ).map( ( _, i ) => i * 0.1 ),
				biases: new Array( outputSize ).fill( 0 ).map( ( _, i ) => i * 0.01 )
			}
		]
	};

}

function makeModel( levels, { emission = false, opacity = false } = {} ) {

	const latentChannels = expectedLatentChannels( levels );
	const decoderInputSize = expectedDecoderInputSize( levels );
	const iblInputSize = expectedIblInputSize( levels );
	const indirectInputSize = expectedIndirectInputSize( levels );

	const model = {
		levels,
		latentGrids: new Array( levels ).fill( 0 ).map( ( _, i ) => makeGrid( 2, 2, CHANNELS_PER_LEVEL, i + 1 ) ),
		rotationWeights: new Array( latentChannels * 12 ).fill( 0 ).map( ( _, i ) => i * 0.001 ),
		decoder: makeMLP( decoderInputSize, 3 ),
		iblHead: makeMLP( iblInputSize, 4 ),
		indirectRadianceHead: makeMLP( indirectInputSize, 3 ),
		indirectIrradianceHead: makeMLP( indirectInputSize, 3 ),
		emissionHead: emission ? makeMLP( latentChannels, 3 ) : null,
		opacityHead: opacity ? makeMLP( latentChannels, 1 ) : null
	};

	return model;

}

describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceManifest', () => {

	describe( 'serializeLayers', () => {

		it( 'copies weights/biases so mutating the model afterwards leaves the serialized layers unchanged', () => {

			const mlp = makeMLP( 3, 2 );
			const layer = mlp.layers[ 0 ];
			const originalWeights = layer.weights.slice();
			const originalBiases = layer.biases.slice();

			const serialized = serializeLayers( mlp );

			// Mutate the model's live arrays in place after serializing.
			layer.weights[ 0 ] = 999;
			layer.biases[ 0 ] = 999;

			expect( serialized[ 0 ].weights ).toEqual( originalWeights );
			expect( serialized[ 0 ].biases ).toEqual( originalBiases );
			expect( serialized[ 0 ].weights[ 0 ] ).not.toBe( 999 );
			expect( serialized[ 0 ].biases[ 0 ] ).not.toBe( 999 );

		} );

		it( 'copies weights/biases so mutating the serialized output afterwards leaves the model unchanged', () => {

			const mlp = makeMLP( 3, 2 );
			const layer = mlp.layers[ 0 ];
			const originalWeights = layer.weights.slice();
			const originalBiases = layer.biases.slice();

			const serialized = serializeLayers( mlp );

			// Mutate the serialized copy in place after serializing.
			serialized[ 0 ].weights[ 0 ] = 999;
			serialized[ 0 ].biases[ 0 ] = 999;

			expect( layer.weights ).toEqual( originalWeights );
			expect( layer.biases ).toEqual( originalBiases );

		} );

		it( 'produces distinct array instances from the model\'s own weights/biases arrays', () => {

			const mlp = makeMLP( 3, 2 );

			const serialized = serializeLayers( mlp );

			expect( serialized[ 0 ].weights ).not.toBe( mlp.layers[ 0 ].weights );
			expect( serialized[ 0 ].biases ).not.toBe( mlp.layers[ 0 ].biases );

		} );

		it( 'carries inputSize/outputSize/activation through unchanged, and preserves layer order', () => {

			const mlp = {
				layers: [
					{ inputSize: 4, outputSize: 8, activation: 'relu', weights: new Array( 32 ).fill( 0 ), biases: new Array( 8 ).fill( 0 ) },
					{ inputSize: 8, outputSize: 3, activation: 'linear', weights: new Array( 24 ).fill( 0 ), biases: new Array( 3 ).fill( 0 ) }
				]
			};

			const serialized = serializeLayers( mlp );

			expect( serialized.length ).toBe( 2 );
			expect( serialized[ 0 ] ).toMatchObject( { inputSize: 4, outputSize: 8, activation: 'relu' } );
			expect( serialized[ 1 ] ).toMatchObject( { inputSize: 8, outputSize: 3, activation: 'linear' } );

		} );

		it( 'round-trips exact weight/bias values through the copy', () => {

			const mlp = makeMLP( 3, 2 );

			const serialized = serializeLayers( mlp );

			expect( serialized[ 0 ].weights ).toEqual( mlp.layers[ 0 ].weights );
			expect( serialized[ 0 ].biases ).toEqual( mlp.layers[ 0 ].biases );

		} );

	} );

	describe( 'createNeuralAppearanceManifest', () => {

		const options = { name: 'test-material', outputActivation: { type: 'linear' } };

		it( 'omits the emission/opacity output blocks when the model has no emissionHead/opacityHead', () => {

			const model = makeModel( 4 );

			const manifest = createNeuralAppearanceManifest( model, options );

			expect( manifest.outputs.emission ).toBeUndefined();
			expect( manifest.outputs.opacity ).toBeUndefined();
			expect( 'emission' in manifest.outputs ).toBe( false );
			expect( 'opacity' in manifest.outputs ).toBe( false );

		} );

		it( 'includes the emission output block only when the model has an emissionHead', () => {

			const model = makeModel( 4, { emission: true } );

			const manifest = createNeuralAppearanceManifest( model, options );

			expect( manifest.outputs.emission ).toBeDefined();
			expect( manifest.outputs.emission.inputSize ).toBe( expectedLatentChannels( 4 ) );
			expect( manifest.outputs.emission.outputActivation ).toEqual( { type: 'linear' } );
			expect( manifest.outputs.opacity ).toBeUndefined();

		} );

		it( 'includes the opacity output block only when the model has an opacityHead, defaulting mode to "mask" and alphaCutoff to 0.5', () => {

			const model = makeModel( 4, { opacity: true } );

			const manifest = createNeuralAppearanceManifest( model, options );

			expect( manifest.outputs.opacity ).toBeDefined();
			expect( manifest.outputs.opacity.inputSize ).toBe( expectedLatentChannels( 4 ) );
			expect( manifest.outputs.opacity.outputActivation ).toEqual( { type: 'sigmoid' } );
			expect( manifest.outputs.opacity.mode ).toBe( 'mask' );
			expect( manifest.outputs.opacity.alphaCutoff ).toBe( 0.5 );
			expect( manifest.outputs.emission ).toBeUndefined();

		} );

		it( 'includes both emission and opacity blocks when the model has both heads', () => {

			const model = makeModel( 4, { emission: true, opacity: true } );

			const manifest = createNeuralAppearanceManifest( model, options );

			expect( manifest.outputs.emission ).toBeDefined();
			expect( manifest.outputs.opacity ).toBeDefined();

		} );

		it( 'honors an explicit opacityMode and alphaCutoff for "mask" mode', () => {

			const model = makeModel( 4, { opacity: true } );

			const manifest = createNeuralAppearanceManifest( model, { ...options, opacityMode: 'mask', alphaCutoff: 0.75 } );

			expect( manifest.outputs.opacity.mode ).toBe( 'mask' );
			expect( manifest.outputs.opacity.alphaCutoff ).toBe( 0.75 );

		} );

		it( 'omits alphaCutoff for a non-"mask" opacityMode unless an explicit finite alphaCutoff is given', () => {

			const model = makeModel( 4, { opacity: true } );

			const manifestWithoutCutoff = createNeuralAppearanceManifest( model, { ...options, opacityMode: 'blend' } );
			expect( manifestWithoutCutoff.outputs.opacity.mode ).toBe( 'blend' );
			expect( 'alphaCutoff' in manifestWithoutCutoff.outputs.opacity ).toBe( false );

			const manifestWithCutoff = createNeuralAppearanceManifest( model, { ...options, opacityMode: 'blend', alphaCutoff: 0.2 } );
			expect( manifestWithCutoff.outputs.opacity.alphaCutoff ).toBe( 0.2 );

		} );

		it( 'sets top-level format/version/name/source and the latents block\'s channelsPerLevel/wrap', () => {

			const model = makeModel( 4 );

			const manifest = createNeuralAppearanceManifest( model, options );

			expect( manifest.format ).toBe( 'three-neural-appearance' );
			expect( Number.isInteger( manifest.version ) ).toBe( true );
			expect( manifest.version ).toBeGreaterThan( 0 );
			expect( manifest.name ).toBe( options.name );
			expect( manifest.source ).toBe( 'THREE.NeuralAppearanceTrainer' );
			expect( manifest.latents.channelsPerLevel ).toBe( CHANNELS_PER_LEVEL );
			expect( manifest.latents.wrap ).toBe( 'repeat' );
			expect( manifest.latents.levels.length ).toBe( model.latentGrids.length );

		} );

		it( 'quantizes each grid\'s data at export time (mutating the model afterwards leaves the manifest\'s already-encoded copy unchanged)', () => {

			const model = makeModel( 2 );
			const grid = model.latentGrids[ 0 ];
			const originalData = grid.data.slice();

			const manifest = createNeuralAppearanceManifest( model, options );

			grid.data[ 0 ] = 12345;

			const level = manifest.latents.levels[ 0 ];
			expect( level.dtype ).toBe( 'uint8' );
			expect( typeof level.dataBase64 ).toBe( 'string' );

			const decoded = decodeUint8Base64( level.dataBase64, level.min, level.max, originalData.length );
			const tolerance = ( level.max - level.min ) / 255 + 1e-6;

			for ( let i = 0; i < originalData.length; i ++ ) {

				expect( Math.abs( decoded[ i ] - originalData[ i ] ) ).toBeLessThanOrEqual( tolerance );

			}

		} );

		it( 'encodes latents.levels[] as uint8 (dtype/min/max/dataBase64) and every output head\'s weights as a float16 mlp block', () => {

			const model = makeModel( 4, { emission: true, opacity: true } );

			const manifest = createNeuralAppearanceManifest( model, options );

			for ( const level of manifest.latents.levels ) {

				expect( level.dtype ).toBe( 'uint8' );
				expect( typeof level.min ).toBe( 'number' );
				expect( typeof level.max ).toBe( 'number' );
				expect( typeof level.dataBase64 ).toBe( 'string' );
				expect( level.data ).toBeUndefined();

			}

			for ( const key of [ 'brdf', 'ibl', 'indirectRadiance', 'indirectIrradiance', 'emission', 'opacity' ] ) {

				const head = manifest.outputs[ key ];
				expect( head.mlp.dtype ).toBe( 'float16' );
				expect( Array.isArray( head.mlp.layout ) ).toBe( true );
				expect( typeof head.mlp.dataBase64 ).toBe( 'string' );
				expect( head.layers ).toBeUndefined();

			}

			expect( manifest.outputs.brdf.rotation.dtype ).toBe( 'float16' );
			expect( typeof manifest.outputs.brdf.rotation.dataBase64 ).toBe( 'string' );
			expect( manifest.outputs.brdf.rotation.weights ).toBeUndefined();

		} );

		describe( 'offset/input-size consistency for non-default `levels`', () => {

			// Expected sizes below come from `expectedXxx` (independently
			// worked out from the documented TBN-frame/IBL/probe layout, see
			// the top-of-file comment) - not from calling
			// NeuralAppearanceFormat.js's own `computeDecoderInputSize` /
			// `computeIblInputSize` / `computeIndirectInputSize` /
			// `computeLatentChannels`. Also cross-checks against a second,
			// completely independent source of truth: each layer object's own
			// `inputSize` field, as written into the *decoder/head MLP given
			// to `makeModel`* before the manifest was ever built - i.e. this
			// also fails if `createNeuralAppearanceManifest` reports an
			// `outputs.*.inputSize` that disagrees with the actual first
			// layer it serialized.
			it.each( [ 2, 3, 4, 5, 6, 8 ] )( 'reports brdf/ibl/indirect inputSize matching both the independently-derived expected size and the model\'s own first-layer inputSize, for levels=%i', ( levels ) => {

				const model = makeModel( levels );

				const manifest = createNeuralAppearanceManifest( model, options );

				expect( manifest.outputs.brdf.inputSize ).toBe( expectedDecoderInputSize( levels ) );
				expect( manifest.outputs.brdf.inputSize ).toBe( model.decoder.layers[ 0 ].inputSize );

				expect( manifest.outputs.brdf.rotation.inputSize ).toBe( expectedLatentChannels( levels ) );
				expect( manifest.outputs.brdf.rotation.inputSize * 12 ).toBe( model.rotationWeights.length );

				expect( manifest.outputs.ibl.inputSize ).toBe( expectedIblInputSize( levels ) );
				expect( manifest.outputs.ibl.inputSize ).toBe( model.iblHead.layers[ 0 ].inputSize );

				expect( manifest.outputs.indirectRadiance.inputSize ).toBe( expectedIndirectInputSize( levels ) );
				expect( manifest.outputs.indirectRadiance.inputSize ).toBe( model.indirectRadianceHead.layers[ 0 ].inputSize );

				expect( manifest.outputs.indirectIrradiance.inputSize ).toBe( expectedIndirectInputSize( levels ) );
				expect( manifest.outputs.indirectIrradiance.inputSize ).toBe( model.indirectIrradianceHead.layers[ 0 ].inputSize );

			} );

			it( 'the exported latents.levels grid count and total channel sum always match model.latentGrids exactly (summed from each grid\'s own `channels` field, not a formula)', () => {

				const model = makeModel( 6 );
				const expectedTotalChannels = model.latentGrids.reduce( ( sum, grid ) => sum + grid.channels, 0 );

				const manifest = createNeuralAppearanceManifest( model, options );

				expect( manifest.latents.levels.length ).toBe( 6 );
				expect( manifest.latents.levels.length ).toBe( model.latentGrids.length );
				expect( manifest.latents.levels.reduce( ( sum, level ) => sum + level.channels, 0 ) ).toBe( expectedTotalChannels );
				expect( expectedTotalChannels ).toBe( 6 * CHANNELS_PER_LEVEL ); // 24 - a concrete, hand-computed number, not just self-consistency

			} );

			it( 'brdf.rotation.outputSize is always the fixed 12-wide TBN-frame constant, independent of levels', () => {

				const model4 = makeModel( 4 );
				const model6 = makeModel( 6 );

				expect( createNeuralAppearanceManifest( model4, options ).outputs.brdf.rotation.outputSize ).toBe( 12 );
				expect( createNeuralAppearanceManifest( model6, options ).outputs.brdf.rotation.outputSize ).toBe( 12 );

			} );

		} );

	} );

} );

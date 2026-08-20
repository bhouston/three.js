import { describe, expect, it } from 'vitest';
import { createModel, forwardDecoderInput, buildIBLInput, buildIndirectProbeInput } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';
import { forwardMLP } from '../../../../examples/jsm/neural/NeuralMLP.js';
import { createNeuralAppearanceManifest } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceManifest.js';
import { evaluateNeuralAppearanceOutputs } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceRuntime.js';
import { LEVELS, CHANNELS_PER_LEVEL, LATENT_CHANNELS, DECODER_INPUT_SIZE } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceFormat.js';

// FIXED. This file documented (and, as of this session, verifies the fix
// for) a real bug found while writing unit tests for NeuralAppearanceModel.js
// - unrelated to the original normal-map/neural-material bug from earlier in
// this session.
//
// `webgpu_materials_neural_appearance.html`'s "grid levels" GUI control lets
// a user pick `levels` from [2, 3, 4, 5, 6, 8] (default 4) and passes it
// straight through to `NeuralAppearanceTrainer` -> `createModel(options,
// random)`, which sizes the actual latent grid pyramid from
// `options.levels` (`resolutions = computeGridLevels(..., options.levels)`,
// one grid per resolution, `CHANNELS_PER_LEVEL` channels each - so the real,
// concatenated latent vector is `options.levels * CHANNELS_PER_LEVEL` wide).
//
// But `NeuralAppearanceFormat.js`'s `LATENT_CHANNELS` (`LEVELS *
// CHANNELS_PER_LEVEL`) - and everything derived from it (`DECODER_INPUT_SIZE
// = LATENT_CHANNELS + 12`, `IBL_INPUT_SIZE`, `INDIRECT_INPUT_SIZE`) - was a
// FIXED module constant computed from the hardcoded default `LEVELS = 4`,
// completely independent of any `options.levels` a caller actually passed.
// `createModel` (and `NeuralAppearanceGPUModel`'s buffer layout, and
// `createNeuralAppearanceManifest`'s exported inputSize metadata, and
// `NeuralAppearanceRuntime.js`'s CPU decoder) all built/read sizes from
// these fixed constants regardless of the model's real `options.levels`.
//
// The result: for any `levels` other than the default 4, the decoder's
// first layer was built expecting `DECODER_INPUT_SIZE` (28, for the default
// LEVELS=4/CHANNELS_PER_LEVEL=4) inputs, but the real assembled decoder
// input was shorter (or longer) - so the forward pass indexed past the end
// of the actual input array (`input[i]` for out-of-range `i` is `undefined`
// in a plain JS array), and `weight * undefined` is `NaN`, which propagated
// through every downstream layer. Every prediction was NaN from the very
// first forward pass - a total, silent training failure for any non-
// default "grid levels" choice, not a subtle accuracy regression.
//
// Fix: `NeuralAppearanceFormat.js` now also exports `computeLatentChannels`/
// `computeDecoderInputSize`/`computeIblInputSize`/`computeIndirectInputSize`
// (functions of `levels`, defaulting to `LEVELS`) - every consumer
// (`NeuralAppearanceModel.js`, `NeuralAppearanceGPUModel.js`,
// `NeuralAppearanceGPUComputeTSL.js`, `NeuralAppearanceManifest.js`,
// `NeuralAppearanceRuntime.js`) now derives its channel/input-size math from
// a model's (or manifest's) actual `levels`/`latents.length`, not the fixed
// constants. The fixed constants themselves are unchanged (still correct
// for the default `levels === LEVELS` case) and still exported for anything
// that only ever uses the default.
describe( 'Addons > Neural > NeuralAppearance > NeuralAppearanceModel (fixed: non-default `levels`)', () => {

	it( 'LATENT_CHANNELS is still a fixed constant (correct for the default), decoupled from any other model\'s configured levels', () => {

		// The constant itself is unchanged - it's every *consumer* that had to
		// stop assuming it always applies.
		expect( LATENT_CHANNELS ).toBe( LEVELS * CHANNELS_PER_LEVEL );

		const random = () => 0.3;
		const model = createModel( { levels: 2, baseResolution: 2, targetResolution: 4, hiddenSize: 4, outputFeatures: {} }, random );
		const actualLatentChannels = model.latentGrids.reduce( ( sum, grid ) => sum + grid.channels, 0 );

		expect( model.levels ).toBe( 2 );
		expect( actualLatentChannels ).toBe( 2 * CHANNELS_PER_LEVEL ); // 8, not LATENT_CHANNELS (16)
		expect( actualLatentChannels ).not.toBe( LATENT_CHANNELS );

		// The decoder is now sized from *this model's* actual 8-wide latents
		// (via computeDecoderInputSize(2)), not the fixed DECODER_INPUT_SIZE.
		expect( model.decoder.layers[ 0 ].inputSize ).toBe( actualLatentChannels + 12 );
		expect( model.decoder.layers[ 0 ].inputSize ).not.toBe( DECODER_INPUT_SIZE );

	} );

	it.each( [ 2, 3, 4, 5, 6, 8 ] )( 'a model built with levels=%i produces finite decoder/IBL/indirect predictions', ( levels ) => {

		const random = () => 0.3;
		const model = createModel( { levels, baseResolution: 2, targetResolution: 4, hiddenSize: 4, outputFeatures: {} }, random );

		const latents = new Array( levels * CHANNELS_PER_LEVEL ).fill( 0 ).map( ( _v, i ) => ( i + 1 ) / 10 );
		const wi = [ 0, 0, 1 ], wo = [ 0.1, 0.2, 0.97 ];

		const decoderInput = forwardDecoderInput( latents, model.rotationWeights, wi, wo ).output;
		const { activations } = forwardMLP( model.decoder, decoderInput );
		const decoderPrediction = activations[ activations.length - 1 ];

		expect( decoderPrediction.every( Number.isFinite ) ).toBe( true );

		const iblInput = buildIBLInput( latents, model.rotationWeights, wo );
		const { activations: iblActivations } = forwardMLP( model.iblHead, iblInput );
		expect( iblActivations[ iblActivations.length - 1 ].every( Number.isFinite ) ).toBe( true );

		const indirectInput = buildIndirectProbeInput( latents, wo, [ 1, 1, 1 ] );
		const { activations: indirectActivations } = forwardMLP( model.indirectRadianceHead, indirectInput );
		expect( indirectActivations[ indirectActivations.length - 1 ].every( Number.isFinite ) ).toBe( true );

	} );

	it.each( [ 2, 3, 4, 5, 6, 8 ] )( 'the full train-model -> export-manifest -> runtime-decode chain produces finite outputs for levels=%i', ( levels ) => {

		let seed = levels * 7919;
		const random = () => {

			seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
			return seed / 0x7fffffff;

		};

		const model = createModel( {
			levels,
			baseResolution: 4,
			targetResolution: 8,
			hiddenSize: 8,
			outputFeatures: { emission: true, opacity: true }
		}, random );

		const manifest = createNeuralAppearanceManifest( model, { outputActivation: { type: 'linear' }, name: 'test', opacityMode: 'mask' } );

		expect( manifest.latents.levels.length ).toBe( levels );
		expect( manifest.outputs.brdf.inputSize ).toBe( levels * CHANNELS_PER_LEVEL + 12 );

		const sample = { uv: [ 0.3, 0.7 ], wi: [ 0.2, 0.1, 0.97 ], wo: [ - 0.1, 0.3, 0.95 ] };
		const outputs = evaluateNeuralAppearanceOutputs( manifest, sample );

		expect( outputs.brdf.every( Number.isFinite ) ).toBe( true );
		expect( outputs.ibl.direction.every( Number.isFinite ) ).toBe( true );
		expect( Number.isFinite( outputs.ibl.roughness ) ).toBe( true );
		expect( outputs.emission.every( Number.isFinite ) ).toBe( true );
		expect( Number.isFinite( outputs.opacity ) ).toBe( true );

	} );

} );

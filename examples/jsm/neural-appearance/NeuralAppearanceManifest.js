// Exported as .neuralAppearance (JSON content, format: 'three-neural-appearance') - see FORMAT/VERSION below

import {
	FORMAT,
	VERSION,
	CHANNELS_PER_LEVEL,
	computeLatentChannels,
	computeDecoderInputSize,
	computeIblInputSize,
	computeIndirectInputSize
} from './NeuralAppearanceFormat.js';
import { normalize } from '../neural/NeuralVectorMath.js';
import {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateNeuralIBLWhiteFurnace,
	integrateNeuralBRDFWhiteFurnace
} from './NeuralAppearanceRuntime.js';
import {
	assignTeacherTargets,
	assignAuxiliaryTeacherTargets,
	normalizeDirectLightingTargets
} from './NeuralAppearanceSampler.js';
import { encodeUint8Base64, encodeFloat16Base64, encodeMLPLayersBase64 } from '../neural/NeuralBinaryCodec.js';
import { computeLatentRanges } from '../neural/NeuralQuantization.js';

/**
 * `createReferenceEvaluations` (below) needs to bilinear-sample the exact
 * float latent values the model was evaluated with (see
 * NeuralAppearanceRuntime.js's `sampleRuntimeLatents`, which reads
 * `level.data[...]` directly) - i.e. a *plain*, uncompacted JSON shape, not
 * the uint8/float16-packed `.neuralAppearance` manifest this module
 * ultimately exports. `createPlainAppearanceJson` builds that intermediate
 * shape (byte-for-byte what `createNeuralAppearanceManifest` produced before
 * the QAT + compact-binary-manifest rework); `compactAppearanceJson` (below)
 * then quantizes/packs it into the real exported manifest. Keeping this as
 * an internal two-step pipeline - rather than teaching
 * NeuralAppearanceRuntime.js's reference evaluator to decode base64 itself -
 * means that evaluator (also used standalone, e.g. by
 * NeuralAppearanceValidator.js) never needs to know the exported manifest
 * is compacted at all.
 */
function createPlainAppearanceJson( model, options ) {

	// Every inputSize below is derived from *this model's own* `model.levels`
	// via the computeXXX helpers, not NeuralAppearanceFormat.js's fixed
	// LATENT_CHANNELS/etc. constants - see that file's doc comment on these
	// helpers for why.
	const latentChannels = computeLatentChannels( model.levels );
	const decoderInputSize = computeDecoderInputSize( model.levels );
	const iblInputSize = computeIblInputSize( model.levels );
	const indirectInputSize = computeIndirectInputSize( model.levels );

	const levels = model.latentGrids.map( ( grid ) => ( {
		width: grid.width,
		height: grid.height,
		channels: grid.channels,
		wrap: 'repeat',
		data: Array.from( grid.data )
	} ) );

	const outputs = {
		brdf: {
			inputSize: decoderInputSize,
			rotation: {
				inputSize: latentChannels,
				outputSize: 12,
				weights: model.rotationWeights.slice()
			},
			layers: serializeLayers( model.decoder ),
			outputActivation: options.outputActivation
		},
		ibl: {
			inputSize: iblInputSize,
			layers: serializeLayers( model.iblHead ),
			outputActivation: { type: 'linear' }
		},
		indirectRadiance: {
			inputSize: indirectInputSize,
			layers: serializeLayers( model.indirectRadianceHead ),
			outputActivation: { type: 'linear' }
		},
		indirectIrradiance: {
			inputSize: indirectInputSize,
			layers: serializeLayers( model.indirectIrradianceHead ),
			outputActivation: { type: 'linear' }
		}
	};

	if ( model.emissionHead ) {

		outputs.emission = {
			inputSize: latentChannels,
			layers: serializeLayers( model.emissionHead ),
			outputActivation: { type: 'linear' }
		};

	}

	if ( model.opacityHead ) {

		const opacityMode = options.opacityMode || 'mask';
		outputs.opacity = {
			inputSize: latentChannels,
			layers: serializeLayers( model.opacityHead ),
			outputActivation: { type: 'sigmoid' },
			mode: opacityMode
		};

		if ( opacityMode === 'mask' ) {

			outputs.opacity.alphaCutoff = Number.isFinite( options.alphaCutoff ) ? options.alphaCutoff : 0.5;

		} else if ( Number.isFinite( options.alphaCutoff ) ) {

			outputs.opacity.alphaCutoff = options.alphaCutoff;

		}

	}

	return {
		format: FORMAT,
		version: VERSION,
		name: options.name,
		source: 'THREE.NeuralAppearanceTrainer',
		latents: {
			levels,
			channelsPerLevel: CHANNELS_PER_LEVEL,
			wrap: 'repeat'
		},
		outputs
	};

}

/**
 * Resolves the per-level `[min, max]` quantization range used to encode the
 * exported latent grid as uint8 - identical policy to
 * NeuralTextureManifest.js's `resolveQuantizationRange`: an explicit
 * `options.quantizationRange`, else `model.quantizationRange` (set by
 * `NeuralAppearanceTrainer.train()` when QAT was actually used - see
 * NeuralQuantization.js), else a plain min/max scan over the model's *final*
 * (post-training) latents via `computeLatentRanges`.
 *
 * As with neural-texture, scanning a range at export time for a model that
 * was never trained with quantization-aware training (QAT) still produces a
 * numerically valid uint8-quantized manifest, but the network's weights
 * were never exposed to the rounding error quantization introduces - so the
 * exported model may reconstruct slightly less accurately than one actually
 * trained with `quantization.mode: 'uint8'` (see NeuralAppearanceTrainer.js's
 * `quantization` option). Prefer training with QAT enabled when export-time
 * compactness matters.
 */
function resolveQuantizationRange( model, options ) {

	if ( options.quantizationRange ) return options.quantizationRange;
	if ( model.quantizationRange ) return model.quantizationRange;

	let offset = 0;
	const gridLevels = [];

	for ( const grid of model.latentGrids ) {

		gridLevels.push( { offset, floatCount: grid.data.length } );
		offset += grid.data.length;

	}

	const flat = new Float32Array( offset );

	for ( let g = 0; g < model.latentGrids.length; g ++ ) {

		flat.set( model.latentGrids[ g ].data, gridLevels[ g ].offset );

	}

	return computeLatentRanges( flat, gridLevels, true );

}

/**
 * Packs one output head's plain (`serializeLayers`-shaped) `layers` array,
 * plus (for `brdf` only) its `rotation` weight matrix, into the compact
 * float16 `mlp`/`rotation` blocks the exported manifest actually stores -
 * every other field on `plainHead` (`inputSize`, `outputActivation`, and for
 * `opacity`, `mode`/`alphaCutoff`) is already plain JSON and carried through
 * unchanged.
 */
function compactHead( plainHead ) {

	const { layers, rotation, ...rest } = plainHead;
	const head = { ...rest, mlp: encodeMLPLayersBase64( layers ) };

	if ( rotation ) {

		head.rotation = {
			inputSize: rotation.inputSize,
			outputSize: rotation.outputSize,
			dtype: 'float16',
			dataBase64: encodeFloat16Base64( Float32Array.from( rotation.weights ) )
		};

	}

	return head;

}

/**
 * Quantizes `plain.latents.levels[]` to uint8 and float16-packs every output
 * head's MLP weights/biases (see `compactHead`), turning the plain JSON
 * `createPlainAppearanceJson` produced into the actual compact
 * `.neuralAppearance` manifest shape `NeuralAppearanceLoader.js` decodes.
 */
function compactAppearanceJson( plain, model, options ) {

	const ranges = resolveQuantizationRange( model, options );

	const levels = plain.latents.levels.map( ( level, index ) => {

		const [ min, max ] = ranges[ index ];

		return {
			width: level.width,
			height: level.height,
			channels: level.channels,
			wrap: level.wrap,
			dtype: 'uint8',
			min,
			max,
			dataBase64: encodeUint8Base64( Float32Array.from( level.data ), min, max )
		};

	} );

	const outputs = {};
	for ( const key of Object.keys( plain.outputs ) ) outputs[ key ] = compactHead( plain.outputs[ key ] );

	return {
		format: plain.format,
		version: plain.version,
		name: plain.name,
		source: plain.source,
		latents: {
			channelsPerLevel: plain.latents.channelsPerLevel,
			wrap: plain.latents.wrap,
			levels
		},
		outputs
	};

}

async function exportNeuralAppearance( model, teacher, options ) {

	const plain = createPlainAppearanceJson( model, options );
	const referenceEvaluations = await createReferenceEvaluations( plain, teacher );

	const json = compactAppearanceJson( plain, model, options );
	json.referenceEvaluations = referenceEvaluations;

	return json;

}

function createNeuralAppearanceManifest( model, options ) {

	const plain = createPlainAppearanceJson( model, options );

	return compactAppearanceJson( plain, model, options );

}

function serializeLayers( mlp ) {

	return mlp.layers.map( ( layer ) => ( {
		inputSize: layer.inputSize,
		outputSize: layer.outputSize,
		activation: layer.activation,
		weights: layer.weights.slice(),
		biases: layer.biases.slice()
	} ) );

}

async function createReferenceEvaluations( json, teacher ) {

	const directions = [
		{ wi: [ 0, 0, 1 ], wo: normalize( [ 0.4, 0.2, 0.894 ] ) },
		{ wi: normalize( [ 0.5, 0.1, 0.86 ] ), wo: normalize( [ - 0.3, 0.4, 0.86 ] ) },
		{ wi: normalize( [ - 0.4, 0.3, 0.866 ] ), wo: [ 0, 0, 1 ] }
	];
	const refs = directions.map( ( direction ) => ( {
		uv: [ 0.5, 0.5 ],
		wi: direction.wi,
		wo: direction.wo,
		normal: [ 0, 0, 1 ],
		tangent: [ 1, 0, 0 ],
		bitangent: [ 0, 1, 0 ],
		encoderInputs: teacher.encodeInputs ? teacher.encodeInputs( [ 0.5, 0.5 ] ) : [ 0.5, 0.5 ]
	} ) );

	await assignTeacherTargets( refs, teacher );
	normalizeDirectLightingTargets( refs );
	await assignAuxiliaryTeacherTargets( refs, teacher );

	// Build a clean, explicitly-whitelisted output object per sample rather
	// than mutating the working `refs` entry and deleting its scratch
	// fields afterward - a whitelist can't leak a field nobody remembered
	// to delete, the way the mutate-then-delete version could.
	return refs.map( ( sample ) => {

		const prediction = evaluateNeuralAppearanceJson( json, sample );
		const outputs = evaluateNeuralAppearanceOutputs( json, sample );

		const result = {
			uv: sample.uv,
			wi: sample.wi,
			wo: sample.wo,
			targetRgb: sample.target.slice(),
			rgb: prediction,
			ibl: outputs.ibl,
			iblWhiteFurnace: evaluateNeuralIBLWhiteFurnace( json, sample ),
			integratedWhiteFurnace: integrateNeuralBRDFWhiteFurnace( json, sample, 32 )
		};

		if ( outputs.indirect ) result.indirect = outputs.indirect;
		if ( outputs.indirectRadiance ) result.indirectRadiance = outputs.indirectRadiance;
		if ( outputs.indirectIrradiance ) result.indirectIrradiance = outputs.indirectIrradiance;

		if ( sample.emissionTarget ) {

			result.targetEmission = sample.emissionTarget.slice();
			result.emission = outputs.emission.slice();

		}

		if ( Number.isFinite( sample.opacityTarget ) ) {

			result.targetOpacity = sample.opacityTarget;
			result.opacity = outputs.opacity;

		}

		return result;

	} );

}

export {
	createNeuralAppearanceManifest,
	serializeLayers,
	exportNeuralAppearance,
	createReferenceEvaluations
};

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

async function exportNeuralAppearance( model, teacher, options ) {

	const json = createNeuralAppearanceManifest( model, options );
	json.referenceEvaluations = await createReferenceEvaluations( json, teacher );

	return json;

}

function createNeuralAppearanceManifest( model, options ) {

	// Every inputSize below is derived from *this model's own* `model.levels`
	// and `model.peOctaves` via the computeXXX helpers, not
	// NeuralAppearanceFormat.js's fixed LATENT_CHANNELS/etc. constants - see
	// that file's doc comment on these helpers for why.
	const peOctaves = model.peOctaves || 0;
	const latentChannels = computeLatentChannels( model.levels );
	const decoderInputSize = computeDecoderInputSize( model.levels, peOctaves );
	const iblInputSize = computeIblInputSize( model.levels, peOctaves );
	const indirectInputSize = computeIndirectInputSize( model.levels, peOctaves );

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
		// Persisted at the manifest root (not per-output) since it's a single
		// model-wide constant, mirroring `latents.levels.length` - every output
		// head's own `inputSize` above already bakes in `peOctaves * 2`, but the
		// runtime/render-time consumers (NeuralAppearanceRuntime.js,
		// NeuralAppearanceNodeMaterial.js) need the raw count too, to know how
		// many tiled-positional-encoding values to compute from a sample's uv
		// and append - see NeuralGridModel.triangleWaveEncode.
		peOctaves,
		outputs
	};

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

import {
	FORMAT,
	VERSION,
	CHANNELS_PER_LEVEL,
	computeLatentChannels,
	computeDecoderInputSize,
	computeIblInputSize,
	computeIndirectInputSize
} from './NeuralAppearanceFormat.js';
import { normalize } from './NeuralAppearanceModel.js';
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
	// (via the computeXXX helpers), not NeuralAppearanceFormat.js's fixed
	// LATENT_CHANNELS/DECODER_INPUT_SIZE/IBL_INPUT_SIZE/INDIRECT_INPUT_SIZE
	// constants - those are only correct when levels === LEVELS (the
	// default). A manifest exported for a non-default `levels` model needs to
	// document its layers' *real* input widths, or a decoder reading this
	// manifest back (NeuralAppearanceRuntime.js) has no way to know how many
	// latent channels it's actually working with. See NeuralAppearanceFormat.
	// js's doc comment on these helpers for the full story.
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

	for ( const sample of refs ) {

		const prediction = evaluateNeuralAppearanceJson( json, sample );
		const outputs = evaluateNeuralAppearanceOutputs( json, sample );
		const iblWhite = evaluateNeuralIBLWhiteFurnace( json, sample );
		const integratedWhite = integrateNeuralBRDFWhiteFurnace( json, sample, 32 );

		sample.targetRgb = sample.target.slice();
		sample.rgb = prediction;
		sample.ibl = outputs.ibl;
		if ( outputs.indirect ) sample.indirect = outputs.indirect;
		if ( outputs.indirectRadiance ) sample.indirectRadiance = outputs.indirectRadiance;
		if ( outputs.indirectIrradiance ) sample.indirectIrradiance = outputs.indirectIrradiance;
		sample.iblWhiteFurnace = iblWhite;
		sample.integratedWhiteFurnace = integratedWhite;
		if ( sample.emissionTarget ) {

			sample.targetEmission = sample.emissionTarget.slice();
			sample.emission = outputs.emission.slice();

		}

		if ( Number.isFinite( sample.opacityTarget ) ) {

			sample.targetOpacity = sample.opacityTarget;
			sample.opacity = outputs.opacity;

		}

		delete sample.normal;
		delete sample.tangent;
		delete sample.bitangent;
		delete sample.encoderInputs;
		delete sample.directTarget;
		delete sample.target;
		delete sample.emissionTarget;
		delete sample.opacityTarget;
		delete sample.weight;

	}

	return refs;

}

export {
	createNeuralAppearanceManifest,
	serializeLayers,
	exportNeuralAppearance,
	createReferenceEvaluations
};

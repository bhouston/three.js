import {
	FORMAT,
	VERSION,
	LATENT_CHANNELS,
	CHANNELS_PER_LEVEL,
	DECODER_INPUT_SIZE,
	IBL_INPUT_SIZE,
	INDIRECT_INPUT_SIZE
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

	const levels = model.latentGrids.map( ( grid ) => ( {
		width: grid.width,
		height: grid.height,
		channels: grid.channels,
		wrap: 'repeat',
		data: Array.from( grid.data )
	} ) );

	const outputs = {
		brdf: {
			inputSize: DECODER_INPUT_SIZE,
			rotation: {
				inputSize: LATENT_CHANNELS,
				outputSize: 12,
				weights: model.rotationWeights.slice()
			},
			layers: serializeLayers( model.decoder ),
			outputActivation: options.outputActivation
		},
		ibl: {
			inputSize: IBL_INPUT_SIZE,
			layers: serializeLayers( model.iblHead ),
			outputActivation: { type: 'linear' }
		},
		indirectRadiance: {
			inputSize: INDIRECT_INPUT_SIZE,
			layers: serializeLayers( model.indirectRadianceHead ),
			outputActivation: { type: 'linear' }
		},
		indirectIrradiance: {
			inputSize: INDIRECT_INPUT_SIZE,
			layers: serializeLayers( model.indirectIrradianceHead ),
			outputActivation: { type: 'linear' }
		}
	};

	if ( model.emissionHead ) {

		outputs.emission = {
			inputSize: LATENT_CHANNELS,
			layers: serializeLayers( model.emissionHead ),
			outputActivation: { type: 'linear' }
		};

	}

	if ( model.opacityHead ) {

		const opacityMode = options.opacityMode || 'mask';
		outputs.opacity = {
			inputSize: LATENT_CHANNELS,
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

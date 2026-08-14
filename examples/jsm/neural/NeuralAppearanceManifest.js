import {
	FORMAT,
	VERSION,
	LATENT_CHANNELS,
	DECODER_INPUT_SIZE,
	IBL_INPUT_SIZE
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

	const mipmaps0 = [];
	const mipmaps1 = [];

	for ( const grid of model.latentGrids ) {

		const data0 = [];
		const data1 = [];

		for ( let y = 0; y < grid.height; y ++ ) {

			for ( let x = 0; x < grid.width; x ++ ) {

				const offset = ( y * grid.width + x ) * LATENT_CHANNELS;
				const latents = grid.data.slice( offset, offset + LATENT_CHANNELS );

				data0.push( latents[ 0 ], latents[ 1 ], latents[ 2 ], latents[ 3 ] );
				data1.push( latents[ 4 ], latents[ 5 ], latents[ 6 ], latents[ 7 ] );

			}

		}

		mipmaps0.push( { width: grid.width, height: grid.height, data: data0 } );
		mipmaps1.push( { width: grid.width, height: grid.height, data: data1 } );

	}

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
			channels: LATENT_CHANNELS,
			wrap: 'repeat',
			textures: [
				{ wrap: 'repeat', mipmaps: mipmaps0 },
				{ wrap: 'repeat', mipmaps: mipmaps1 }
			]
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
		duvDx: [ 1 / 1024, 0 ],
		duvDy: [ 0, 1 / 1024 ],
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

		sample.mip = 0;
		sample.targetRgb = sample.target.slice();
		sample.rgb = prediction;
		sample.ibl = outputs.ibl;
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
		delete sample.duvDx;
		delete sample.duvDy;
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

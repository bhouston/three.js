import { DataUtils } from 'three';
import { createGpuMaterialTeacher } from './NeuralAppearanceTeacherEvaluator.js';

const FORMAT = 'three-neural-appearance';
const VERSION = 1;
const LATENT_CHANNELS = 8;
const DECODER_INPUT_SIZE = 20;
const DEFAULT_OPTIONS = {
	resolution: 8,
	iterations: 2000,
	batchSize: 1024,
	learningRate: 0.001,
	cosineAnnealingScale: 0.01,
	seed: 1,
	hiddenSize: 16,
	yieldEvery: 8,
	colorAugmentation: false,
	minimumTrainingCosine: 0.05,
	maxGradientNorm: 1,
	previewSampleCount: 64,
	outputActivation: { type: 'linear' },
	name: 'trained neural appearance'
};

/**
 * Browser-side trainer that distills an opaque MaterialX-loaded
 * MeshPhysicalNodeMaterial into the compact neural appearance runtime format.
 *
 * This first version mirrors the BSDF encoding phase from NVIDIA's training
 * pipeline with ordinary JavaScript math so it can run in examples without
 * additional ML dependencies. The public contract is the exported
 * `three-neural-appearance` JSON consumed by `NeuralAppearanceLoader`.
 *
 * @three_import import { NeuralAppearanceTrainer } from 'three/addons/materials/NeuralAppearanceTrainer.js';
 */
class NeuralAppearanceTrainer {

	constructor( options = {} ) {

		this.options = { ...DEFAULT_OPTIONS, ...options };
		this.random = createRandom( this.options.seed );

	}

	async train( { material, renderer = null, onProgress = null, ...options } = {} ) {

		const settings = { ...this.options, ...options };
		validateTrainingSettings( settings );
		const teacher = options.teacher || createGpuMaterialTeacher( material, renderer, settings );
		const model = createModel( settings, this.random );
		let validationSamples = null;
		let directionalValidationSamples = null;
		let lastLoss = Infinity;
		let validationLoss = Infinity;
		let validation = null;

		if ( renderer && renderer.isWebGPURenderer === true && renderer.init ) {

			await renderer.init();

		}

		if ( teacher.init ) await teacher.init();
		validationSamples = await generateTrainingSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ), colorAugmentation: false }, teacher, createRandom( settings.seed + 0x9e3779b9 ), settings.iterations );
		directionalValidationSamples = await generateValidationSamples( { ...settings, batchSize: Math.min( 64, settings.batchSize ) }, teacher );

		for ( let iteration = 0; iteration < settings.iterations; iteration ++ ) {

			const lr = getLearningRate( settings, iteration );
			const samples = await generateTrainingSamples( settings, teacher, this.random, iteration );
			lastLoss = trainBatch( model, samples, teacher, lr, iteration + 1, settings.maxGradientNorm );
			const manifest = createNeuralAppearanceManifest( model, settings );
			validation = evaluateRuntimeValidation( manifest, validationSamples, settings.previewSampleCount );
			validation.directional = evaluateRuntimeValidation( manifest, directionalValidationSamples, 0 );
			validationLoss = validation.loss;

			if ( onProgress ) {

				onProgress( {
					iteration: iteration + 1,
					iterations: settings.iterations,
					loss: lastLoss,
					validationLoss,
					validation,
					learningRate: lr
				} );

			}

			if ( settings.yieldEvery > 0 && iteration % settings.yieldEvery === settings.yieldEvery - 1 ) {

				await yieldToBrowser();

			}

		}

		const json = await exportNeuralAppearance( model, teacher, settings );

		return {
			json,
			loss: lastLoss,
			validationLoss,
			validation,
			model,
			teacher
		};

	}

}

function validateTrainingSettings( settings ) {

	if ( Number.isInteger( settings.resolution ) === false || settings.resolution < 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: resolution must be a positive integer.' );

	}

	if ( settings.outputActivation === null || settings.outputActivation === undefined || settings.outputActivation.type !== 'linear' ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Only linear output activation is supported during training.' );

	}

	if ( Number.isFinite( settings.maxGradientNorm ) === false || settings.maxGradientNorm <= 0 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: maxGradientNorm must be finite and greater than zero.' );

	}

	if ( Number.isFinite( settings.minimumTrainingCosine ) === false || settings.minimumTrainingCosine < 0 || settings.minimumTrainingCosine > 1 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: minimumTrainingCosine must be between zero and one.' );

	}

}

async function generateTrainingSamples( options, teacher, random, iteration = 0 ) {

	const batchSize = options.batchSize;
	const gridSize = Math.max( 1, Math.floor( Math.sqrt( batchSize ) ) );
	const samples = [];
	const mipLevelCount = getMipLevelCount( options.resolution, options.resolution );
	const augmentationRatio = options.colorAugmentation && teacher.supportsColorAugmentation === true ? 0.5 * ( 1 + Math.cos( Math.PI * Math.min( iteration / Math.max( 1, options.iterations ), 1 ) ) ) : 0;

	for ( let i = 0; i < batchSize; i ++ ) {

		const x = i % gridSize;
		const y = Math.floor( i / gridSize ) % gridSize;
		const uv = [
			( x + random() ) / gridSize,
			( y + random() ) / gridSize
		];
		const directions = sampleTeacherDirections( random );

		for ( let mip = 0; mip < mipLevelCount; mip ++ ) {

			const footprint = Math.pow( 2, mip ) / Math.max( 1, options.resolution );
			samples.push( {
				uv: uv.slice(),
				wi: directions.wi.slice(),
				wo: directions.wo.slice(),
				normal: [ 0, 0, 1 ],
				tangent: [ 1, 0, 0 ],
				bitangent: [ 0, 1, 0 ],
				duvDx: [ footprint, 0 ],
				duvDy: [ 0, footprint ],
				mip,
				encoderInputs: teacher.encodeInputs( uv )
			} );

		}

	}

	await assignTeacherTargets( samples, teacher );
	normalizeDirectLightingTargets( samples, options.minimumTrainingCosine );

	for ( const sample of samples ) {

		if ( random() < augmentationRatio ) {

			augmentColorChannels( sample, random );

		}

	}

	return samples;

}

async function generateValidationSamples( options, teacher ) {

	const sampleCount = options.batchSize;
	const gridSize = Math.max( 1, Math.ceil( Math.sqrt( sampleCount ) ) );
	const mipLevelCount = getMipLevelCount( options.resolution, options.resolution );
	const cosines = [ 0.025, 0.1, 0.4, 0.8 ];
	const samples = [];

	for ( let i = 0; i < sampleCount; i ++ ) {

		const uv = [
			( i % gridSize + 0.5 ) / gridSize,
			( Math.floor( i / gridSize ) + 0.5 ) / gridSize
		];
		const wiCosine = cosines[ i % cosines.length ];
		const woCosine = cosines[ Math.floor( i / cosines.length ) % cosines.length ];
		const azimuth = 2 * Math.PI * ( i + 0.5 ) / sampleCount;
		const wi = directionFromCosine( wiCosine, azimuth );
		const wo = directionFromCosine( woCosine, azimuth * 1.61803398875 );

		for ( let mip = 0; mip < mipLevelCount; mip ++ ) {

			const footprint = Math.pow( 2, mip ) / Math.max( 1, options.resolution );
			samples.push( {
				uv: uv.slice(),
				wi: wi.slice(),
				wo: wo.slice(),
				normal: [ 0, 0, 1 ],
				tangent: [ 1, 0, 0 ],
				bitangent: [ 0, 1, 0 ],
				duvDx: [ footprint, 0 ],
				duvDy: [ 0, footprint ],
				mip,
				encoderInputs: teacher.encodeInputs( uv )
			} );

		}

	}

	await assignTeacherTargets( samples, teacher );
	normalizeDirectLightingTargets( samples, options.minimumTrainingCosine );

	return samples;

}

function createModel( options, random ) {

	const decoder = createMLP( DECODER_INPUT_SIZE, [ options.hiddenSize, options.hiddenSize ], 3, random, 'relu', 'linear' );
	const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 );
	const latentGrids = createLatentMipGrids( options.resolution, options.resolution, random );

	return {
		decoder,
		rotationWeights,
		rotationGrad: new Array( rotationWeights.length ).fill( 0 ),
		rotationM: new Array( rotationWeights.length ).fill( 0 ),
		rotationV: new Array( rotationWeights.length ).fill( 0 ),
		latentGrid: latentGrids[ 0 ],
		latentGrids
	};

}

function createLatentMipGrids( baseWidth, baseHeight, random ) {

	const grids = [];
	let width = baseWidth;
	let height = baseHeight;

	while ( true ) {

		grids.push( createLatentGrid( width, height, random ) );

		if ( width === 1 && height === 1 ) break;
		width = Math.max( 1, width >> 1 );
		height = Math.max( 1, height >> 1 );

	}

	return grids;

}

function createLatentGrid( width, height, random ) {

	const data = new Array( width * height * LATENT_CHANNELS );

	for ( let i = 0; i < data.length; i ++ ) {

		data[ i ] = ( random() * 2 - 1 ) * 0.25;

	}

	return {
		width,
		height,
		data,
		grad: new Array( data.length ).fill( 0 ),
		m: new Array( data.length ).fill( 0 ),
		v: new Array( data.length ).fill( 0 )
	};

}

function createMLP( inputSize, hiddenLayers, outputSize, random, hiddenActivation, outputActivation ) {

	const sizes = [ inputSize, ...hiddenLayers, outputSize ];
	const layers = [];

	for ( let i = 0; i < sizes.length - 1; i ++ ) {

		const input = sizes[ i ];
		const output = sizes[ i + 1 ];
		const scale = Math.sqrt( 2 / input );
		const weights = new Array( input * output );
		const biases = new Array( output ).fill( 0 );

		for ( let j = 0; j < weights.length; j ++ ) {

			weights[ j ] = ( random() * 2 - 1 ) * scale;

		}

		layers.push( {
			inputSize: input,
			outputSize: output,
			weights,
			biases,
			activation: i === sizes.length - 2 ? outputActivation : hiddenActivation,
			mWeights: new Array( weights.length ).fill( 0 ),
			vWeights: new Array( weights.length ).fill( 0 ),
			mBiases: new Array( biases.length ).fill( 0 ),
			vBiases: new Array( biases.length ).fill( 0 )
		} );

	}

	return { layers };

}

function trainBatch( model, samples, teacher, learningRate, step, maxGradientNorm ) {

	const sampleWeightSum = getSampleWeightSum( samples );
	if ( sampleWeightSum <= 0 ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Batch contains no finite, well-conditioned teacher samples.' );

	}

	const invBatch = 1 / sampleWeightSum;
	let loss = 0;

	zeroGradients( model.decoder );
	for ( const grid of model.latentGrids ) zeroLatentGradients( grid );
	model.rotationGrad.fill( 0 );

	for ( const sample of samples ) {

		const latentGrid = model.latentGrids[ sample.mip || 0 ];
		const latentRun = sampleLatents( latentGrid, sample.uv );
		const latents = latentRun.output;
		const inputRun = forwardDecoderInput( latents, model.rotationWeights, sample.wi, sample.wo );
		const decoderRun = forwardMLP( model.decoder, inputRun.output );
		const prediction = decoderRun.output.map( ( value ) => Math.max( 0, value ) );
		const target = sample.target;
		const sampleWeight = sample.weight !== undefined ? sample.weight : 1;
		const gradPrediction = [ 0, 0, 0 ];

		if ( sampleWeight === 0 ) continue;
		if ( prediction.every( Number.isFinite ) === false || target.every( Number.isFinite ) === false ) {

			throw new Error( 'THREE.NeuralAppearanceTrainer: Encountered a non-finite training sample.' );

		}

		for ( let i = 0; i < 3; i ++ ) {

			const pred = Math.max( prediction[ i ], 1e-6 );
			const ref = Math.max( target[ i ], 1e-6 );
			const predLog = powerLog( pred, 3 );
			const refLog = powerLog( ref, 3 );
			const diff = predLog - refLog;
			loss += Math.abs( diff ) * sampleWeight * invBatch / 3;
			gradPrediction[ i ] = Math.sign( diff ) * Math.pow( pred, 1 / 3 - 1 ) * sampleWeight * invBatch / 3;

		}

		const gradDecoderInput = backwardMLP( model.decoder, decoderRun, gradPrediction );
		const inputGradients = backwardDecoderInput( inputRun, gradDecoderInput, model.rotationWeights );

		for ( let i = 0; i < model.rotationGrad.length; i ++ ) {

			model.rotationGrad[ i ] += inputGradients.rotationWeights[ i ];

		}

		scatterLatentGradients( latentGrid, latentRun, inputGradients.latents );

	}

	clipModelGradients( model, maxGradientNorm );
	for ( const grid of model.latentGrids ) applyAdamLatents( grid, learningRate, step );
	applyAdamRotation( model, learningRate, step );
	applyAdam( model.decoder, learningRate, step );
	assertModelFinite( model );

	if ( Number.isFinite( loss ) === false ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Training produced a non-finite loss.' );

	}

	return loss;

}

function clipModelGradients( model, maxNorm ) {

	if ( ! Number.isFinite( maxNorm ) || maxNorm <= 0 ) return;

	const gradients = [ model.rotationGrad ];

	for ( const grid of model.latentGrids ) gradients.push( grid.grad );
	for ( const layer of model.decoder.layers ) gradients.push( layer.gradWeights, layer.gradBiases );

	let squaredNorm = 0;

	for ( const values of gradients ) {

		for ( const value of values ) squaredNorm += value * value;

	}

	const norm = Math.sqrt( squaredNorm );
	if ( ! Number.isFinite( norm ) ) {

		throw new Error( 'THREE.NeuralAppearanceTrainer: Training produced non-finite gradients.' );

	}

	if ( norm <= maxNorm ) return;

	const scale = maxNorm / norm;

	for ( const values of gradients ) {

		for ( let i = 0; i < values.length; i ++ ) values[ i ] *= scale;

	}

}

function assertModelFinite( model ) {

	assertFiniteArray( model.rotationWeights, 'learned-frame weights' );

	for ( const grid of model.latentGrids ) assertFiniteArray( grid.data, 'latent values' );

	for ( const layer of model.decoder.layers ) {

		assertFiniteArray( layer.weights, 'decoder weights' );
		assertFiniteArray( layer.biases, 'decoder biases' );

	}

}

function assertFiniteArray( values, label ) {

	if ( values.every( Number.isFinite ) === false ) {

		throw new Error( `THREE.NeuralAppearanceTrainer: Training produced non-finite ${ label }.` );

	}

}

function evaluateRuntimeValidation( json, samples, previewSampleCount = DEFAULT_OPTIONS.previewSampleCount ) {

	const invBatch = 1 / Math.max( getSampleWeightSum( samples ), 1 );
	let loss = 0;
	const angularBins = {
		wi: createAngularBins(),
		wo: createAngularBins()
	};
	const preview = createRuntimePreview( previewSampleCount, samples.length );

	for ( const sample of samples ) {

		const sampleWeight = sample.weight !== undefined ? sample.weight : 1;
		const prediction = evaluateNeuralAppearanceJson( json, sample );
		const target = sample.target;
		const nDotL = Math.max( sample.wi[ 2 ], 0 );
		const directTarget = Array.isArray( sample.directTarget ) ? sample.directTarget : target.map( ( value ) => value * nDotL );
		const directPrediction = prediction.map( ( value ) => value * nDotL );

		if ( sampleWeight > 0 ) {

			if ( prediction.every( Number.isFinite ) === false || target.every( Number.isFinite ) === false ) {

				throw new Error( 'THREE.NeuralAppearanceTrainer: Runtime validation produced non-finite values.' );

			}

			for ( let i = 0; i < 3; i ++ ) {

				const pred = Math.max( prediction[ i ], 1e-6 );
				const ref = Math.max( target[ i ], 1e-6 );
				const diff = powerLog( pred, 3 ) - powerLog( ref, 3 );
				loss += Math.abs( diff ) * sampleWeight * invBatch / 3;

			}

		}

		if ( Array.isArray( sample.directTarget ) && sample.wi[ 2 ] >= 0 && sample.wo[ 2 ] >= 0 ) {

			accumulateAngularError( angularBins.wi, sample.wi[ 2 ], directPrediction, sample.directTarget );
			accumulateAngularError( angularBins.wo, sample.wo[ 2 ], directPrediction, sample.directTarget );

		}

		appendRuntimePreviewSample( preview, sample, directTarget, directPrediction, sampleWeight );

	}

	return {
		loss,
		sampleCount: samples.length,
		mipLevels: json.latents.textures[ 0 ].mipmaps.length,
		preview,
		angularBins: {
			wi: finalizeAngularBins( angularBins.wi ),
			wo: finalizeAngularBins( angularBins.wo )
		}
	};

}

function createRuntimePreview( maxSampleCount, sourceSampleCount ) {

	const sampleCount = Math.min( Math.max( 0, Math.floor( maxSampleCount ) ), sourceSampleCount );
	const width = sampleCount > 0 ? Math.ceil( Math.sqrt( sampleCount ) ) : 0;

	return {
		width,
		height: width > 0 ? Math.ceil( sampleCount / width ) : 0,
		samples: []
	};

}

function appendRuntimePreviewSample( preview, sample, targetRgb, predictionRgb, weight ) {

	if ( preview.samples.length >= preview.width * preview.height ) return;

	preview.samples.push( {
		uv: sample.uv.slice( 0, 2 ),
		wi: sample.wi.slice( 0, 3 ),
		wo: sample.wo.slice( 0, 3 ),
		mip: sample.mip || 0,
		weight,
		targetRgb: sanitizeRgb( targetRgb ),
		predictionRgb: sanitizeRgb( predictionRgb )
	} );

}

function sanitizeRgb( rgb ) {

	return [
		sanitizeChannel( rgb[ 0 ] ),
		sanitizeChannel( rgb[ 1 ] ),
		sanitizeChannel( rgb[ 2 ] )
	];

}

function sanitizeChannel( value ) {

	return Number.isFinite( value ) ? Math.max( value, 0 ) : 0;

}

function createAngularBins() {

	return [
		{ min: 0, max: 0.05, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 },
		{ min: 0.05, max: 0.2, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 },
		{ min: 0.2, max: 1, count: 0, absoluteError: 0, targetMagnitude: 0, maxChannelError: 0 }
	];

}

function accumulateAngularError( bins, cosine, prediction, target ) {

	const bin = bins.find( ( candidate, index ) => cosine >= candidate.min && ( cosine < candidate.max || index === bins.length - 1 ) );
	if ( bin === undefined || target.every( Number.isFinite ) === false ) return;

	bin.count ++;

	for ( let channel = 0; channel < 3; channel ++ ) {

		const error = Math.abs( prediction[ channel ] - target[ channel ] );
		bin.absoluteError += error;
		bin.targetMagnitude += Math.abs( target[ channel ] );
		bin.maxChannelError = Math.max( bin.maxChannelError, error );

	}

}

function finalizeAngularBins( bins ) {

	return bins.map( ( bin ) => ( {
		min: bin.min,
		max: bin.max,
		count: bin.count,
		meanAbsoluteError: bin.count > 0 ? bin.absoluteError / ( bin.count * 3 ) : null,
		relativeAbsoluteError: bin.targetMagnitude > 0 ? bin.absoluteError / bin.targetMagnitude : null,
		maxChannelError: bin.count > 0 ? bin.maxChannelError : null
	} ) );

}

function getSampleWeightSum( samples ) {

	let weight = 0;

	for ( const sample of samples ) {

		weight += sample.weight !== undefined ? sample.weight : 1;

	}

	return weight;

}

function forwardMLP( mlp, input ) {

	const activations = [ input.slice() ];
	const preActivations = [];
	let values = input.slice();

	for ( const layer of mlp.layers ) {

		const next = new Array( layer.outputSize );
		const pre = new Array( layer.outputSize );

		for ( let output = 0; output < layer.outputSize; output ++ ) {

			let value = layer.biases[ output ];

			for ( let inputIndex = 0; inputIndex < layer.inputSize; inputIndex ++ ) {

				value += layer.weights[ output * layer.inputSize + inputIndex ] * values[ inputIndex ];

			}

			pre[ output ] = value;
			next[ output ] = activate( value, layer.activation );

		}

		preActivations.push( pre );
		activations.push( next );
		values = next;

	}

	return { activations, preActivations, output: values };

}

function backwardMLP( mlp, run, gradOutput ) {

	let grad = gradOutput.slice();

	for ( let layerIndex = mlp.layers.length - 1; layerIndex >= 0; layerIndex -- ) {

		const layer = mlp.layers[ layerIndex ];
		const input = run.activations[ layerIndex ];
		const pre = run.preActivations[ layerIndex ];
		const gradInput = new Array( layer.inputSize ).fill( 0 );

		if ( layer.gradWeights === undefined ) {

			layer.gradWeights = new Array( layer.weights.length ).fill( 0 );
			layer.gradBiases = new Array( layer.biases.length ).fill( 0 );

		}

		for ( let output = 0; output < layer.outputSize; output ++ ) {

			const delta = grad[ output ] * activateDerivative( pre[ output ], layer.activation );
			layer.gradBiases[ output ] += delta;

			for ( let inputIndex = 0; inputIndex < layer.inputSize; inputIndex ++ ) {

				const weightIndex = output * layer.inputSize + inputIndex;
				layer.gradWeights[ weightIndex ] += delta * input[ inputIndex ];
				gradInput[ inputIndex ] += delta * layer.weights[ weightIndex ];

			}

		}

		grad = gradInput;

	}

	return grad;

}

function zeroGradients( mlp ) {

	for ( const layer of mlp.layers ) {

		layer.gradWeights = new Array( layer.weights.length ).fill( 0 );
		layer.gradBiases = new Array( layer.biases.length ).fill( 0 );

	}

}

function applyAdam( mlp, learningRate, step ) {

	const beta1 = 0.9;
	const beta2 = 0.999;
	const epsilon = 1e-7;
	const beta1Correction = 1 - Math.pow( beta1, step );
	const beta2Correction = 1 - Math.pow( beta2, step );

	for ( const layer of mlp.layers ) {

		for ( let i = 0; i < layer.weights.length; i ++ ) {

			const grad = layer.gradWeights[ i ];
			layer.mWeights[ i ] = beta1 * layer.mWeights[ i ] + ( 1 - beta1 ) * grad;
			layer.vWeights[ i ] = beta2 * layer.vWeights[ i ] + ( 1 - beta2 ) * grad * grad;
			layer.weights[ i ] -= learningRate * ( layer.mWeights[ i ] / beta1Correction ) / ( Math.sqrt( layer.vWeights[ i ] / beta2Correction ) + epsilon );

		}

		for ( let i = 0; i < layer.biases.length; i ++ ) {

			const grad = layer.gradBiases[ i ];
			layer.mBiases[ i ] = beta1 * layer.mBiases[ i ] + ( 1 - beta1 ) * grad;
			layer.vBiases[ i ] = beta2 * layer.vBiases[ i ] + ( 1 - beta2 ) * grad * grad;
			layer.biases[ i ] -= learningRate * ( layer.mBiases[ i ] / beta1Correction ) / ( Math.sqrt( layer.vBiases[ i ] / beta2Correction ) + epsilon );

		}

	}

}

function sampleLatents( grid, uv ) {

	const x = uv[ 0 ] * grid.width - 0.5;
	const y = uv[ 1 ] * grid.height - 0.5;
	const x0 = Math.floor( x );
	const y0 = Math.floor( y );
	const tx = x - x0;
	const ty = y - y0;
	const taps = [
		{ x: wrapIndex( x0, grid.width ), y: wrapIndex( y0, grid.height ), weight: ( 1 - tx ) * ( 1 - ty ) },
		{ x: wrapIndex( x0 + 1, grid.width ), y: wrapIndex( y0, grid.height ), weight: tx * ( 1 - ty ) },
		{ x: wrapIndex( x0, grid.width ), y: wrapIndex( y0 + 1, grid.height ), weight: ( 1 - tx ) * ty },
		{ x: wrapIndex( x0 + 1, grid.width ), y: wrapIndex( y0 + 1, grid.height ), weight: tx * ty }
	];
	const output = new Array( LATENT_CHANNELS ).fill( 0 );

	for ( const tap of taps ) {

		const offset = ( tap.y * grid.width + tap.x ) * LATENT_CHANNELS;

		for ( let channel = 0; channel < LATENT_CHANNELS; channel ++ ) {

			output[ channel ] += grid.data[ offset + channel ] * tap.weight;

		}

	}

	return { output, taps };

}

function scatterLatentGradients( grid, latentRun, gradLatents ) {

	for ( const tap of latentRun.taps ) {

		const offset = ( tap.y * grid.width + tap.x ) * LATENT_CHANNELS;

		for ( let channel = 0; channel < LATENT_CHANNELS; channel ++ ) {

			grid.grad[ offset + channel ] += gradLatents[ channel ] * tap.weight;

		}

	}

}

function zeroLatentGradients( grid ) {

	grid.grad.fill( 0 );

}

function applyAdamLatents( grid, learningRate, step ) {

	const beta1 = 0.9;
	const beta2 = 0.999;
	const epsilon = 1e-7;
	const beta1Correction = 1 - Math.pow( beta1, step );
	const beta2Correction = 1 - Math.pow( beta2, step );

	for ( let i = 0; i < grid.data.length; i ++ ) {

		const grad = grid.grad[ i ];
		grid.m[ i ] = beta1 * grid.m[ i ] + ( 1 - beta1 ) * grad;
		grid.v[ i ] = beta2 * grid.v[ i ] + ( 1 - beta2 ) * grad * grad;
		grid.data[ i ] -= learningRate * ( grid.m[ i ] / beta1Correction ) / ( Math.sqrt( grid.v[ i ] / beta2Correction ) + epsilon );

	}

}

function applyAdamRotation( model, learningRate, step ) {

	const beta1 = 0.9;
	const beta2 = 0.999;
	const epsilon = 1e-7;
	const beta1Correction = 1 - Math.pow( beta1, step );
	const beta2Correction = 1 - Math.pow( beta2, step );

	for ( let i = 0; i < model.rotationWeights.length; i ++ ) {

		const grad = model.rotationGrad[ i ];
		model.rotationM[ i ] = beta1 * model.rotationM[ i ] + ( 1 - beta1 ) * grad;
		model.rotationV[ i ] = beta2 * model.rotationV[ i ] + ( 1 - beta2 ) * grad * grad;
		model.rotationWeights[ i ] -= learningRate * ( model.rotationM[ i ] / beta1Correction ) / ( Math.sqrt( model.rotationV[ i ] / beta2Correction ) + epsilon );

	}

}

function wrapIndex( value, size ) {

	return ( ( value % size ) + size ) % size;

}

function getMipLevelCount( width, height ) {

	let levels = 1;

	while ( width > 1 || height > 1 ) {

		width = Math.max( 1, width >> 1 );
		height = Math.max( 1, height >> 1 );
		levels ++;

	}

	return levels;

}

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
		decoder: {
			inputSize: DECODER_INPUT_SIZE,
			rotation: {
				inputSize: LATENT_CHANNELS,
				outputSize: 12,
				weights: model.rotationWeights.slice()
			},
			layers: model.decoder.layers.map( ( layer, index ) => ( {
				inputSize: layer.inputSize,
				outputSize: layer.outputSize,
				activation: index === model.decoder.layers.length - 1 ? 'linear' : 'relu',
				weights: layer.weights.slice(),
				biases: layer.biases.slice()
			} ) ),
			outputActivation: options.outputActivation
		}
	};

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
		encoderInputs: teacher.encodeInputs( [ 0.5, 0.5 ] )
	} ) );

	await assignTeacherTargets( refs, teacher );
	normalizeDirectLightingTargets( refs );

	for ( const sample of refs ) {

		const prediction = evaluateNeuralAppearanceJson( json, sample );

		sample.mip = 0;
		sample.targetRgb = sample.target.slice();
		sample.rgb = prediction;
		delete sample.normal;
		delete sample.tangent;
		delete sample.bitangent;
		delete sample.duvDx;
		delete sample.duvDy;
		delete sample.encoderInputs;
		delete sample.directTarget;
		delete sample.target;
		delete sample.weight;

	}

	return refs;

}

async function assignTeacherTargets( samples, teacher ) {

	const targets = teacher.evaluateBatch ? await teacher.evaluateBatch( samples ) : await Promise.all( samples.map( ( sample ) => teacher.evaluate( sample ) ) );

	for ( let i = 0; i < samples.length; i ++ ) {

		samples[ i ].target = targets[ i ];

	}

}

function normalizeDirectLightingTargets( samples, minimumCosine = DEFAULT_OPTIONS.minimumTrainingCosine ) {

	for ( const sample of samples ) {

		const nDotL = Math.max( sample.wi[ 2 ], 0 );
		const validTarget = sample.target.length >= 3 && sample.target.every( Number.isFinite );
		sample.directTarget = validTarget ? sample.target.slice( 0, 3 ) : null;

		if ( validTarget && nDotL >= minimumCosine ) {

			sample.target = sample.target.map( ( value ) => value / nDotL );
			sample.weight = nDotL;

		} else {

			sample.target = [ 0, 0, 0 ];
			sample.weight = 0;

		}

	}

}

function evaluateNeuralAppearanceJson( json, reference ) {

	const mip = selectRuntimeMipLevel( json, reference );
	const latents = sampleRuntimeLatents( json, reference.uv || [ 0.5, 0.5 ], mip );
	const wi = normalize( reference.wi );
	const wo = normalize( reference.wo );
	const input = buildDecoderInput( latents, json.decoder.rotation.weights, wi, wo );

	return evaluateDecoderLayers( json.decoder.layers, input, json.decoder.outputActivation );

}

function selectRuntimeMipLevel( json, reference ) {

	const mipmaps = json.latents.textures[ 0 ].mipmaps;
	const maxMip = mipmaps.length - 1;

	if ( reference.duvDx && reference.duvDy ) {

		const base = mipmaps[ 0 ];
		const dx = Math.hypot( reference.duvDx[ 0 ] * base.width, reference.duvDx[ 1 ] * base.height );
		const dy = Math.hypot( reference.duvDy[ 0 ] * base.width, reference.duvDy[ 1 ] * base.height );
		const computed = Math.min( Math.max( Math.log2( Math.max( dx, dy, 1 ) ), 0 ), maxMip );

		return Math.floor( computed + 0.5 );

	}

	return Math.min( Math.max( Math.round( reference.mip || 0 ), 0 ), maxMip );

}

function sampleRuntimeLatents( json, uv, mipLevel ) {

	const textures = json.latents.textures;
	const mipmap = textures[ 0 ].mipmaps[ mipLevel ];
	const x = uv[ 0 ] * mipmap.width - 0.5;
	const y = uv[ 1 ] * mipmap.height - 0.5;
	const x0 = Math.floor( x );
	const y0 = Math.floor( y );
	const tx = x - x0;
	const ty = y - y0;
	const taps = [
		{ x: x0, y: y0, weight: ( 1 - tx ) * ( 1 - ty ) },
		{ x: x0 + 1, y: y0, weight: tx * ( 1 - ty ) },
		{ x: x0, y: y0 + 1, weight: ( 1 - tx ) * ty },
		{ x: x0 + 1, y: y0 + 1, weight: tx * ty }
	];
	const latents = new Array( LATENT_CHANNELS ).fill( 0 );

	for ( let textureIndex = 0; textureIndex < textures.length; textureIndex ++ ) {

		const texture = textures[ textureIndex ];
		const level = texture.mipmaps[ mipLevel ];
		const repeat = texture.wrap === 'repeat';

		for ( const tap of taps ) {

			const tapX = repeat ? wrapIndex( tap.x, level.width ) : Math.min( Math.max( tap.x, 0 ), level.width - 1 );
			const tapY = repeat ? wrapIndex( tap.y, level.height ) : Math.min( Math.max( tap.y, 0 ), level.height - 1 );
			const offset = ( tapY * level.width + tapX ) * 4;

			for ( let channel = 0; channel < 4; channel ++ ) {

				const value = DataUtils.fromHalfFloat( DataUtils.toHalfFloat( level.data[ offset + channel ] ) );
				latents[ textureIndex * 4 + channel ] += value * tap.weight;

			}

		}

	}

	return latents;

}

function evaluateDecoderLayers( layers, input, outputActivation = { type: 'linear' } ) {

	let values = input.slice();

	for ( const layer of layers ) {

		const next = [];

		for ( let output = 0; output < layer.outputSize; output ++ ) {

			let value = layer.biases[ output ];

			for ( let inputIndex = 0; inputIndex < layer.inputSize; inputIndex ++ ) {

				value += layer.weights[ output * layer.inputSize + inputIndex ] * values[ inputIndex ];

			}

			next.push( layer.activation === 'relu' ? Math.max( 0, value ) : value );

		}

		values = next;

	}

	if ( outputActivation.type === 'scaledSigmoid' ) {

		const scale = outputActivation.scale !== undefined ? outputActivation.scale : 1;
		return values.map( ( value ) => scale / ( 1 + Math.exp( - value ) ) );

	}

	if ( outputActivation.type === 'exp' ) {

		const offset = outputActivation.offset || 0;
		return values.map( ( value ) => Math.exp( value + offset ) );

	}

	return values.map( ( value ) => Math.max( 0, value ) );

}

function buildDecoderInput( latents, rotationWeights, wi, wo ) {

	return forwardDecoderInput( latents, rotationWeights, wi, wo ).output;

}

function forwardDecoderInput( latents, rotationWeights, wi, wo ) {

	const output = latents.slice();
	const frames = [];

	for ( let frame = 0; frame < 2; frame ++ ) {

		const offset = frame * 6;
		const rawN = [
			linearRotationValue( latents, rotationWeights, offset ),
			linearRotationValue( latents, rotationWeights, offset + 1 ),
			linearRotationValue( latents, rotationWeights, offset + 2 ) + 1
		];
		const rawT = [
			linearRotationValue( latents, rotationWeights, offset + 3 ) + 1,
			linearRotationValue( latents, rotationWeights, offset + 4 ),
			linearRotationValue( latents, rotationWeights, offset + 5 )
		];
		const n = normalize( rawN );
		const t = normalize( rawT );
		const b = cross( n, t );

		output.push( dot( wi, t ), dot( wi, b ), dot( wi, n ) );
		output.push( dot( wo, t ), dot( wo, b ), dot( wo, n ) );
		frames.push( { offset, rawN, rawT, n, t } );

	}

	return { output, latents, wi, wo, frames };

}

function backwardDecoderInput( run, gradOutput, rotationWeights ) {

	const gradLatents = gradOutput.slice( 0, LATENT_CHANNELS );
	const gradRotationWeights = new Array( rotationWeights.length ).fill( 0 );

	for ( let frame = 0; frame < run.frames.length; frame ++ ) {

		const frameRun = run.frames[ frame ];
		const inputOffset = LATENT_CHANNELS + frame * 6;
		const gradT = addScaledVectors( run.wi, gradOutput[ inputOffset ], run.wo, gradOutput[ inputOffset + 3 ] );
		const gradB = addScaledVectors( run.wi, gradOutput[ inputOffset + 1 ], run.wo, gradOutput[ inputOffset + 4 ] );
		const gradN = addScaledVectors( run.wi, gradOutput[ inputOffset + 2 ], run.wo, gradOutput[ inputOffset + 5 ] );
		const gradNormalizedN = addVectors( gradN, cross( frameRun.t, gradB ) );
		const gradNormalizedT = addVectors( gradT, cross( gradB, frameRun.n ) );
		const gradRawN = backwardNormalize( frameRun.rawN, frameRun.n, gradNormalizedN );
		const gradRawT = backwardNormalize( frameRun.rawT, frameRun.t, gradNormalizedT );
		const gradFrame = [ ...gradRawN, ...gradRawT ];

		for ( let outputIndex = 0; outputIndex < 6; outputIndex ++ ) {

			const rotationOutput = frameRun.offset + outputIndex;
			const weightOffset = rotationOutput * LATENT_CHANNELS;

			for ( let latentIndex = 0; latentIndex < LATENT_CHANNELS; latentIndex ++ ) {

				gradRotationWeights[ weightOffset + latentIndex ] += gradFrame[ outputIndex ] * run.latents[ latentIndex ];
				gradLatents[ latentIndex ] += gradFrame[ outputIndex ] * rotationWeights[ weightOffset + latentIndex ];

			}

		}

	}

	return {
		latents: gradLatents,
		rotationWeights: gradRotationWeights
	};

}

function backwardNormalize( raw, normalized, gradNormalized ) {

	const inverseLength = 1 / ( Math.hypot( raw[ 0 ], raw[ 1 ], raw[ 2 ] ) || 1 );
	const projectedGradient = dot( normalized, gradNormalized );

	return [
		( gradNormalized[ 0 ] - normalized[ 0 ] * projectedGradient ) * inverseLength,
		( gradNormalized[ 1 ] - normalized[ 1 ] * projectedGradient ) * inverseLength,
		( gradNormalized[ 2 ] - normalized[ 2 ] * projectedGradient ) * inverseLength
	];

}

function addVectors( a, b ) {

	return [ a[ 0 ] + b[ 0 ], a[ 1 ] + b[ 1 ], a[ 2 ] + b[ 2 ] ];

}

function addScaledVectors( a, scaleA, b, scaleB ) {

	return [
		a[ 0 ] * scaleA + b[ 0 ] * scaleB,
		a[ 1 ] * scaleA + b[ 1 ] * scaleB,
		a[ 2 ] * scaleA + b[ 2 ] * scaleB
	];

}

function linearRotationValue( latents, weights, outputIndex ) {

	let value = 0;

	for ( let i = 0; i < latents.length; i ++ ) {

		value += weights[ outputIndex * LATENT_CHANNELS + i ] * latents[ i ];

	}

	return value;

}

function sampleTeacherDirections( random ) {

	const mode = random();

	if ( mode < 0.6 ) {

		return {
			wi: sampleHemisphereUniform( random ),
			wo: sampleHemisphereUniform( random )
		};

	}

	if ( mode < 0.8 ) {

		return {
			wi: sampleHemisphereCosine( random ),
			wo: sampleHemisphereCosine( random )
		};

	}

	if ( mode < 0.9 ) {

		return {
			wi: sampleSphereUniform( random ),
			wo: sampleHemisphereUniform( random )
		};

	}

	return sampleRusinkiewicz( random );

}

function sampleRusinkiewicz( random ) {

	for ( let attempts = 0; attempts < 32; attempts ++ ) {

		const wh = sampleHemisphereUniform( random );
		const wd = sampleHemisphereUniform( random );
		const wo = reflect( [ - wd[ 0 ], - wd[ 1 ], - wd[ 2 ] ], wh );
		const wi = reflect( wd, wh );

		if ( wi[ 2 ] > 1e-5 && wo[ 2 ] >= 0 ) return { wi, wo };

	}

	return { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] };

}

function sampleSphereUniform( random ) {

	const z = random() * 2 - 1;
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

}

function sampleHemisphereUniform( random ) {

	const z = random();
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

}

function sampleHemisphereCosine( random ) {

	const z = Math.sqrt( random() );
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

}

function directionFromCosine( cosine, azimuth ) {

	const radius = Math.sqrt( Math.max( 0, 1 - cosine * cosine ) );
	return [ radius * Math.cos( azimuth ), radius * Math.sin( azimuth ), cosine ];

}

function reflect( vector, normal ) {

	const scale = 2 * dot( vector, normal );

	return normalize( [
		scale * normal[ 0 ] - vector[ 0 ],
		scale * normal[ 1 ] - vector[ 1 ],
		scale * normal[ 2 ] - vector[ 2 ]
	] );

}

function augmentColorChannels( sample, random ) {

	const pattern = [
		Math.floor( random() * 3 ),
		Math.floor( random() * 3 ),
		Math.floor( random() * 3 )
	];
	const albedoOffset = 3;
	const f0Offset = 8;
	const target = sample.target.slice();
	const albedo = sample.encoderInputs.slice( albedoOffset, albedoOffset + 3 );
	const f0 = sample.encoderInputs.slice( f0Offset, f0Offset + 3 );

	for ( let i = 0; i < 3; i ++ ) {

		sample.target[ i ] = target[ pattern[ i ] ];
		sample.encoderInputs[ albedoOffset + i ] = albedo[ pattern[ i ] ];
		sample.encoderInputs[ f0Offset + i ] = f0[ pattern[ i ] ];

	}

}

function activate( value, activation ) {

	if ( activation === 'relu' ) return Math.max( 0, value );
	if ( activation === 'leakyRelu' ) return value >= 0 ? value : value * 0.01;
	if ( activation === 'tanh' ) return Math.tanh( value );

	return value;

}

function activateDerivative( value, activation ) {

	if ( activation === 'relu' ) return value > 0 ? 1 : 0;
	if ( activation === 'leakyRelu' ) return value >= 0 ? 1 : 0.01;
	if ( activation === 'tanh' ) {

		const tanh = Math.tanh( value );
		return 1 - tanh * tanh;

	}

	return 1;

}

function powerLog( value, power ) {

	return power * ( Math.pow( value, 1 / power ) - 1 );

}

function getLearningRate( options, iteration ) {

	const t = Math.min( iteration / Math.max( 1, options.iterations - 1 ), 1 );
	const cosine = 0.5 * ( 1 + Math.cos( Math.PI * t ) );
	const scale = options.cosineAnnealingScale + cosine * ( 1 - options.cosineAnnealingScale );

	return options.learningRate * scale;

}

function dot( a, b ) {

	return a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];

}

function cross( a, b ) {

	return [
		a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ],
		a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ],
		a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ]
	];

}

function normalize( value ) {

	const length = Math.hypot( value[ 0 ], value[ 1 ], value[ 2 ] ) || 1;

	return [ value[ 0 ] / length, value[ 1 ] / length, value[ 2 ] / length ];

}

function createRandom( seed ) {

	let state = seed >>> 0;

	return function random() {

		state = ( state + 0x6D2B79F5 ) | 0;
		let value = Math.imul( state ^ state >>> 15, 1 | state );
		value ^= value + Math.imul( value ^ value >>> 7, 61 | value );

		return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

	};

}

function yieldToBrowser() {

	return new Promise( ( resolve ) => setTimeout( resolve, 0 ) );

}

export {
	NeuralAppearanceTrainer,
	createGpuMaterialTeacher,
	evaluateNeuralAppearanceJson,
	evaluateRuntimeValidation,
	generateTrainingSamples,
	normalizeDirectLightingTargets,
	exportNeuralAppearance
};

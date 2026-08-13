import { Color } from 'three';

const FORMAT = 'three-neural-appearance';
const VERSION = 1;
const LATENT_CHANNELS = 8;
const DECODER_INPUT_SIZE = 20;
const DEFAULT_ENCODER_INPUT_SIZE = 14;
const DEFAULT_OPTIONS = {
	resolution: 8,
	iterations: 2000,
	batchSize: 1024,
	learningRate: 0.001,
	cosineAnnealingScale: 0.01,
	seed: 1,
	hiddenSize: 16,
	yieldEvery: 8,
	colorAugmentation: true,
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
		const teacher = options.teacher || createPhysicalMaterialTeacher( material );
		const model = createModel( settings, this.random );
		let lastLoss = Infinity;

		if ( renderer && renderer.isWebGPURenderer === true && renderer.init ) {

			await renderer.init();

		}

		for ( let iteration = 0; iteration < settings.iterations; iteration ++ ) {

			const lr = getLearningRate( settings, iteration );
			const samples = generateTrainingSamples( settings, teacher, this.random, iteration );
			lastLoss = trainBatch( model, samples, teacher, lr, iteration + 1 );

			if ( onProgress ) {

				onProgress( {
					iteration: iteration + 1,
					iterations: settings.iterations,
					loss: lastLoss,
					learningRate: lr
				} );

			}

			if ( settings.yieldEvery > 0 && iteration % settings.yieldEvery === settings.yieldEvery - 1 ) {

				await yieldToBrowser();

			}

		}

		const json = exportNeuralAppearance( model, teacher, settings );

		return {
			json,
			loss: lastLoss,
			model,
			teacher
		};

	}

}

function createPhysicalMaterialTeacher( material = {} ) {

	const baseColor = readColorValue( material.colorNode, material.color || 0xffffff );
	const roughness = readNumberValue( material.roughnessNode, material.roughness !== undefined ? material.roughness : 0.5 );
	const metalness = readNumberValue( material.metalnessNode, material.metalness !== undefined ? material.metalness : 0 );
	const specularIntensity = readNumberValue( material.specularIntensityNode, material.specularIntensity !== undefined ? material.specularIntensity : 1 );
	const specularColor = readColorValue( material.specularColorNode, material.specularColor || 0xffffff );
	const ior = readNumberValue( material.iorNode, material.ior !== undefined ? material.ior : 1.5 );
	const clearcoat = readNumberValue( material.clearcoatNode, material.clearcoat !== undefined ? material.clearcoat : 0 );
	const clearcoatRoughness = readNumberValue( material.clearcoatRoughnessNode, material.clearcoatRoughness !== undefined ? material.clearcoatRoughness : 0.1 );

	if ( readNumberValue( material.transmissionNode, material.transmission || 0 ) > 0 ) {

		console.warn( 'THREE.NeuralAppearanceTrainer: Transmission is ignored by the opaque neural appearance runtime.' );

	}

	return {
		baseColor,
		roughness: clamp( roughness, 0.04, 1 ),
		metalness: clamp( metalness, 0, 1 ),
		specularColor,
		specularIntensity: clamp( specularIntensity, 0, 1 ),
		ior,
		clearcoat: clamp( clearcoat, 0, 1 ),
		clearcoatRoughness: clamp( clearcoatRoughness, 0.04, 1 ),

		evaluate( sample ) {

			return evaluatePhysicalBRDF( this, sample.wi, sample.wo );

		},

		encodeInputs() {

			const dielectricF0 = Math.pow( ( this.ior - 1 ) / ( this.ior + 1 ), 2 ) * this.specularIntensity;
			const f0 = [
				lerp( dielectricF0 * this.specularColor[ 0 ], this.baseColor[ 0 ], this.metalness ),
				lerp( dielectricF0 * this.specularColor[ 1 ], this.baseColor[ 1 ], this.metalness ),
				lerp( dielectricF0 * this.specularColor[ 2 ], this.baseColor[ 2 ], this.metalness )
			];

			return [
				0, 0, 1,
				this.baseColor[ 0 ], this.baseColor[ 1 ], this.baseColor[ 2 ],
				this.roughness,
				this.metalness,
				f0[ 0 ], f0[ 1 ], f0[ 2 ],
				this.clearcoat,
				this.clearcoatRoughness,
				0
			];

		}
	};

}

function generateTrainingSamples( options, teacher, random, iteration = 0 ) {

	const batchSize = options.batchSize;
	const gridSize = Math.max( 1, Math.floor( Math.sqrt( batchSize ) ) );
	const samples = [];
	const augmentationRatio = options.colorAugmentation ? 0.5 * ( 1 + Math.cos( Math.PI * Math.min( iteration / Math.max( 1, options.iterations ), 1 ) ) ) : 0;

	for ( let i = 0; i < batchSize; i ++ ) {

		const x = i % gridSize;
		const y = Math.floor( i / gridSize ) % gridSize;
		const uv = [
			( x + random() ) / gridSize,
			( y + random() ) / gridSize
		];
		const directions = random() < 0.95 ? sampleRusinkiewicz( random ) : sampleWiGgxWo( teacher.roughness, random );
		const sample = {
			uv,
			wi: directions.wi,
			wo: directions.wo,
			encoderInputs: teacher.encodeInputs()
		};

		sample.target = teacher.evaluate( sample );

		if ( random() < augmentationRatio ) {

			augmentColorChannels( sample, random );

		}

		samples.push( sample );

	}

	return samples;

}

function createModel( options, random ) {

	const encoder = createMLP( DEFAULT_ENCODER_INPUT_SIZE, [ 64, 64, 64 ], LATENT_CHANNELS, random, 'leakyRelu', 'tanh' );
	const decoder = createMLP( DECODER_INPUT_SIZE, [ options.hiddenSize, options.hiddenSize ], 3, random, 'relu', 'linear' );
	const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 );

	return {
		encoder,
		decoder,
		rotationWeights
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

function trainBatch( model, samples, teacher, learningRate, step ) {

	const invBatch = 1 / samples.length;
	let loss = 0;

	zeroGradients( model.encoder );
	zeroGradients( model.decoder );

	for ( const sample of samples ) {

		const encoderRun = forwardMLP( model.encoder, sample.encoderInputs );
		const latents = encoderRun.output;
		const decoderInput = buildDecoderInput( latents, sample.wi, sample.wo );
		const decoderRun = forwardMLP( model.decoder, decoderInput );
		const prediction = decoderRun.output.map( ( value ) => Math.max( 0, value ) );
		const target = sample.target;
		const gradPrediction = [ 0, 0, 0 ];

		for ( let i = 0; i < 3; i ++ ) {

			const pred = Math.max( prediction[ i ], 1e-6 );
			const ref = Math.max( target[ i ], 1e-6 );
			const predLog = powerLog( pred, 3 );
			const refLog = powerLog( ref, 3 );
			const diff = predLog - refLog;
			loss += Math.abs( diff ) * invBatch / 3;
			gradPrediction[ i ] = Math.sign( diff ) * Math.pow( pred, 1 / 3 - 1 ) * invBatch / 3;

			if ( decoderRun.output[ i ] <= 0 ) {

				gradPrediction[ i ] = 0;

			}

		}

		const gradDecoderInput = backwardMLP( model.decoder, decoderRun, gradPrediction );
		backwardMLP( model.encoder, encoderRun, gradDecoderInput.slice( 0, LATENT_CHANNELS ) );

	}

	applyAdam( model.encoder, learningRate, step );
	applyAdam( model.decoder, learningRate, step );

	return loss;

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

function exportNeuralAppearance( model, teacher, options ) {

	const mipmaps0 = [];
	const mipmaps1 = [];
	let width = options.resolution;
	let height = options.resolution;

	while ( width >= 1 && height >= 1 ) {

		const data0 = [];
		const data1 = [];

		for ( let y = 0; y < height; y ++ ) {

			for ( let x = 0; x < width; x ++ ) {

				const inputs = teacher.encodeInputs( [ ( x + 0.5 ) / width, ( y + 0.5 ) / height ] );
				const latents = forwardMLP( model.encoder, inputs ).output;

				data0.push( latents[ 0 ], latents[ 1 ], latents[ 2 ], latents[ 3 ] );
				data1.push( latents[ 4 ], latents[ 5 ], latents[ 6 ], latents[ 7 ] );

			}

		}

		mipmaps0.push( { width, height, data: data0 } );
		mipmaps1.push( { width, height, data: data1 } );

		if ( width === 1 && height === 1 ) break;
		width = Math.max( 1, width >> 1 );
		height = Math.max( 1, height >> 1 );

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
		},
		referenceEvaluations: createReferenceEvaluations( model, teacher )
	};

}

function createReferenceEvaluations( model, teacher ) {

	const refs = [];
	const directions = [
		{ wi: [ 0, 0, 1 ], wo: normalize( [ 0.4, 0.2, 0.894 ] ) },
		{ wi: normalize( [ 0.5, 0.1, 0.86 ] ), wo: normalize( [ - 0.3, 0.4, 0.86 ] ) },
		{ wi: normalize( [ - 0.4, 0.3, 0.866 ] ), wo: [ 0, 0, 1 ] }
	];

	for ( const direction of directions ) {

		const sample = {
			uv: [ 0.5, 0.5 ],
			wi: direction.wi,
			wo: direction.wo,
			encoderInputs: teacher.encodeInputs()
		};

		const latent = forwardMLP( model.encoder, sample.encoderInputs ).output;
		const prediction = forwardMLP( model.decoder, buildDecoderInput( latent, sample.wi, sample.wo ) ).output;

		refs.push( {
			uv: sample.uv,
			wi: sample.wi,
			wo: sample.wo,
			mip: 0,
			rgb: prediction.map( ( value ) => Math.max( 0, value ) )
		} );

	}

	return refs;

}

function buildDecoderInput( latents, wi, wo ) {

	return [
		...latents,
		wi[ 0 ], wi[ 1 ], wi[ 2 ],
		wo[ 0 ], wo[ 1 ], wo[ 2 ],
		wi[ 0 ], wi[ 1 ], wi[ 2 ],
		wo[ 0 ], wo[ 1 ], wo[ 2 ]
	];

}

function evaluatePhysicalBRDF( teacher, wi, wo ) {

	if ( wi[ 2 ] <= 1e-5 || wo[ 2 ] < 0 ) return [ 0, 0, 0 ];

	const diffuseColor = teacher.baseColor.map( ( channel ) => channel * ( 1 - teacher.metalness ) );
	const dielectricF0 = Math.pow( ( teacher.ior - 1 ) / ( teacher.ior + 1 ), 2 ) * teacher.specularIntensity;
	const f0 = [
		lerp( dielectricF0 * teacher.specularColor[ 0 ], teacher.baseColor[ 0 ], teacher.metalness ),
		lerp( dielectricF0 * teacher.specularColor[ 1 ], teacher.baseColor[ 1 ], teacher.metalness ),
		lerp( dielectricF0 * teacher.specularColor[ 2 ], teacher.baseColor[ 2 ], teacher.metalness )
	];
	const half = normalize( [ wi[ 0 ] + wo[ 0 ], wi[ 1 ] + wo[ 1 ], wi[ 2 ] + wo[ 2 ] ] );
	const dotLH = clamp( dot( wi, half ), 0, 1 );
	const fresnel = f0.map( ( value ) => value + ( 1 - value ) * Math.pow( 1 - dotLH, 5 ) );
	const ggx = ggxBRDF( teacher.roughness, wi, wo, half );
	const nDotL = clamp( wi[ 2 ], 0, 1 );
	const color = new Array( 3 );

	for ( let i = 0; i < 3; i ++ ) {

		color[ i ] = nDotL * ( diffuseColor[ i ] / Math.PI * ( 1 - fresnel[ i ] ) + fresnel[ i ] * ggx );

	}

	if ( teacher.clearcoat > 0 ) {

		const coatF = 0.04 + 0.96 * Math.pow( 1 - dotLH, 5 );
		const coat = teacher.clearcoat * coatF * ggxBRDF( teacher.clearcoatRoughness, wi, wo, half );

		for ( let i = 0; i < 3; i ++ ) {

			color[ i ] += nDotL * coat;

		}

	}

	return color.map( ( value ) => Math.max( 0, value ) );

}

function ggxBRDF( roughness, wi, wo, half ) {

	const alpha = Math.max( 0.001, roughness * roughness );
	const dotNL = clamp( wi[ 2 ], 0, 1 );
	const dotNV = clamp( wo[ 2 ], 0, 1 );
	const dotNH = clamp( half[ 2 ], 0, 1 );
	const alpha2 = alpha * alpha;
	const denom = dotNH * dotNH * ( alpha2 - 1 ) + 1;
	const d = alpha2 / ( Math.PI * denom * denom );
	const v = smithG( dotNL, alpha ) * smithG( dotNV, alpha ) / Math.max( 4 * dotNL * dotNV, 1e-6 );

	return d * v;

}

function smithG( dotNV, alpha ) {

	const alpha2 = alpha * alpha;
	const dot2 = dotNV * dotNV;

	return 2 * dotNV / Math.max( dotNV + Math.sqrt( alpha2 + ( 1 - alpha2 ) * dot2 ), 1e-6 );

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

function sampleWiGgxWo( roughness, random ) {

	const wi = sampleHemisphereUniform( random );
	const alpha = Math.max( 0.001, roughness * roughness );
	const phi = 2 * Math.PI * random();
	const u = random();
	const cosTheta = Math.sqrt( ( 1 - u ) / ( 1 + ( alpha * alpha - 1 ) * u ) );
	const sinTheta = Math.sqrt( Math.max( 0, 1 - cosTheta * cosTheta ) );
	const half = [ Math.cos( phi ) * sinTheta, Math.sin( phi ) * sinTheta, cosTheta ];
	const wo = reflect( [ - wi[ 0 ], - wi[ 1 ], - wi[ 2 ] ], half );

	return wi[ 2 ] > 1e-5 && wo[ 2 ] >= 0 ? { wi, wo } : { wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] };

}

function sampleHemisphereUniform( random ) {

	const z = random();
	const phi = 2 * Math.PI * random();
	const r = Math.sqrt( Math.max( 0, 1 - z * z ) );

	return [ r * Math.cos( phi ), r * Math.sin( phi ), z ];

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

function readNumberValue( node, fallback ) {

	if ( typeof node === 'number' ) return node;

	const value = readNodeConstantValue( node );
	if ( typeof value === 'number' ) return value;

	return fallback;

}

function readColorValue( node, fallback ) {

	const value = readNodeConstantValue( node );
	if ( value ) return colorToArray( value );

	return colorToArray( fallback );

}

function readNodeConstantValue( node, depth = 0 ) {

	if ( node === undefined || node === null || depth > 8 ) return undefined;
	if ( ( node.isConstNode === true || node.isUniformNode === true ) && node.value !== undefined ) return node.value;

	return readNodeConstantValue( node.node, depth + 1 );

}

function colorToArray( value ) {

	if ( Array.isArray( value ) ) return value.slice( 0, 3 );
	if ( value && value.isColor === true ) return [ value.r, value.g, value.b ];
	if ( typeof value === 'number' ) {

		const color = new Color( value );
		return [ color.r, color.g, color.b ];

	}

	if ( value && value.r !== undefined && value.g !== undefined && value.b !== undefined ) return [ value.r, value.g, value.b ];

	return [ 1, 1, 1 ];

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

function normalize( value ) {

	const length = Math.hypot( value[ 0 ], value[ 1 ], value[ 2 ] ) || 1;

	return [ value[ 0 ] / length, value[ 1 ] / length, value[ 2 ] / length ];

}

function lerp( a, b, t ) {

	return a + ( b - a ) * t;

}

function clamp( value, min, max ) {

	return Math.min( Math.max( value, min ), max );

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
	createPhysicalMaterialTeacher,
	generateTrainingSamples,
	evaluatePhysicalBRDF,
	exportNeuralAppearance
};

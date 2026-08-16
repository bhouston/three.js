import {
	LATENT_CHANNELS,
	DECODER_INPUT_SIZE,
	IBL_INPUT_SIZE,
	IBL_OUTPUT_SIZE,
	INDIRECT_INPUT_SIZE,
	INDIRECT_OUTPUT_SIZE
} from './NeuralAppearanceFormat.js';
import {
	createMLP,
	forwardMLP
} from './NeuralAppearanceMLP.js';

const LATENT_INIT_SCALE = 0.35;

function createModel( options, random ) {

	const decoder = createMLP( DECODER_INPUT_SIZE, [ options.hiddenSize, options.hiddenSize ], 3, random, 'relu', 'linear' );
	const iblHiddenSize = options.iblHiddenSize || Math.min( Math.max( options.hiddenSize || 32, 16 ), 32 );
	const iblHead = createMLP( IBL_INPUT_SIZE, [ iblHiddenSize ], IBL_OUTPUT_SIZE, random, 'relu', 'linear' );
	const indirectRadianceHead = createMLP( INDIRECT_INPUT_SIZE, [ iblHiddenSize ], INDIRECT_OUTPUT_SIZE, random, 'relu', 'linear' );
	const indirectIrradianceHead = createMLP( INDIRECT_INPUT_SIZE, [ iblHiddenSize ], INDIRECT_OUTPUT_SIZE, random, 'relu', 'linear' );
	const emissionHead = options.outputFeatures && options.outputFeatures.emission ?
		createMLP( LATENT_CHANNELS, [], 3, random, 'relu', 'linear' ) :
		null;
	const opacityHead = options.outputFeatures && options.outputFeatures.opacity ?
		createMLP( LATENT_CHANNELS, [], 1, random, 'relu', 'linear' ) :
		null;
	const rotationWeights = new Array( LATENT_CHANNELS * 12 ).fill( 0 );
	const latentGrids = createLatentMipGrids( options.resolution, options.resolution, random );
	const model = {
		decoder,
		iblHead,
		indirectRadianceHead,
		indirectIrradianceHead,
		emissionHead,
		opacityHead,
		rotationWeights,
		latentGrid: latentGrids[ 0 ],
		latentGrids
	};

	decorrelateLinearRgbHead( model );

	return model;

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

		data[ i ] = ( random() * 2 - 1 ) * LATENT_INIT_SCALE;

	}

	return {
		width,
		height,
		data
	};

}

function decorrelateLinearRgbHead( model ) {

	const layer = model.decoder.layers[ model.decoder.layers.length - 1 ];
	if ( layer.outputSize !== 3 || model.latentGrid === undefined ) return;

	const hiddenSize = layer.inputSize;
	const probes = [
		{ uv: [ 0.2, 0.2 ], wi: [ 0, 0, 1 ], wo: [ 0, 0, 1 ] },
		{ uv: [ 0.8, 0.2 ], wi: [ 0.5, 0, 0.866 ], wo: [ 0, 0.5, 0.866 ] },
		{ uv: [ 0.2, 0.8 ], wi: [ - 0.4, 0.3, 0.866 ], wo: [ 0.2, - 0.3, 0.93 ] },
		{ uv: [ 0.8, 0.8 ], wi: [ 0.2, 0.6, 0.77 ], wo: [ - 0.5, 0.2, 0.84 ] },
		{ uv: [ 0.5, 0.5 ], wi: [ 0, 0.7, 0.71 ], wo: [ 0.7, 0, 0.71 ] }
	];
	const hiddenMean = new Array( hiddenSize ).fill( 0 );

	for ( const probe of probes ) {

		const latents = sampleLatents( model.latentGrid, probe.uv ).output;
		const input = forwardDecoderInput( latents, model.rotationWeights, probe.wi, probe.wo ).output;
		const activations = forwardMLP( model.decoder, input ).activations;
		const hidden = activations[ activations.length - 2 ];

		for ( let i = 0; i < hiddenSize; i ++ ) {

			hiddenMean[ i ] += hidden[ i ] / probes.length;

		}

	}

	let hiddenNormSquared = 0;

	for ( let i = 0; i < hiddenSize; i ++ ) {

		hiddenNormSquared += hiddenMean[ i ] * hiddenMean[ i ];

	}

	if ( hiddenNormSquared < 1e-12 ) return;

	for ( let channel = 0; channel < 3; channel ++ ) {

		let dot = 0;

		for ( let i = 0; i < hiddenSize; i ++ ) {

			dot += layer.weights[ channel * hiddenSize + i ] * hiddenMean[ i ];

		}

		const projection = dot / hiddenNormSquared;

		for ( let i = 0; i < hiddenSize; i ++ ) {

			layer.weights[ channel * hiddenSize + i ] -= projection * hiddenMean[ i ];

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
		const rawB = cross( n, t );
		const b = normalize( rawB );

		output.push( dot( wi, t ), dot( wi, b ), dot( wi, n ) );
		output.push( dot( wo, t ), dot( wo, b ), dot( wo, n ) );
		frames.push( { offset, rawN, rawT, rawB, n, t, b } );

	}

	return { output, latents, wi, wo, frames };

}

function buildDecoderInput( latents, rotationWeights, wi, wo ) {

	return forwardDecoderInput( latents, rotationWeights, wi, wo ).output;

}

function forwardIBLInput( latents, rotationWeights, wo ) {

	const decoderInput = forwardDecoderInput( latents, rotationWeights, [ 0, 0, 1 ], wo );
	const output = latents.slice();

	for ( const frame of decoderInput.frames ) {

		output.push( dot( wo, frame.t ), dot( wo, frame.b ), dot( wo, frame.n ) );

	}

	return { output, latents, wo, frames: decoderInput.frames };

}

function buildIBLInput( latents, rotationWeights, wo ) {

	const input = forwardIBLInput( latents, rotationWeights, wo ).output;

	if ( input.length !== IBL_INPUT_SIZE ) {

		throw new Error( `THREE.NeuralAppearanceModel: IBL input has ${ input.length } values, expected ${ IBL_INPUT_SIZE }.` );

	}

	return input;

}

function buildIndirectProbeInput( latents, wo, probe ) {

	const input = latents.slice();
	const incoming = probe || [ 1, 1, 1 ];

	input.push( wo[ 0 ], wo[ 1 ], wo[ 2 ] );
	input.push( incoming[ 0 ], incoming[ 1 ], incoming[ 2 ] );

	if ( input.length !== INDIRECT_INPUT_SIZE ) {

		throw new Error( `THREE.NeuralAppearanceModel: Indirect probe input has ${ input.length } values, expected ${ INDIRECT_INPUT_SIZE }.` );

	}

	return input;

}

function unpackIBLOutput( values ) {

	if ( values.length !== IBL_OUTPUT_SIZE ) {

		throw new Error( `THREE.NeuralAppearanceModel: IBL query output has ${ values.length } values, expected ${ IBL_OUTPUT_SIZE }.` );

	}

	return {
		direction: normalize( values.slice( 0, 3 ) ),
		roughness: sigmoid( values[ 3 ] )
	};

}

function sigmoid( value ) {

	return 1 / ( 1 + Math.exp( - value ) );

}

function linearRotationValue( latents, weights, outputIndex ) {

	let value = 0;

	for ( let i = 0; i < latents.length; i ++ ) {

		value += weights[ outputIndex * LATENT_CHANNELS + i ] * latents[ i ];

	}

	return value;

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

function wrapIndex( value, size ) {

	return ( ( value % size ) + size ) % size;

}

function getSampleWeightSum( samples ) {

	let sum = 0;

	for ( const sample of samples ) {

		sum += sample.weight !== undefined ? sample.weight : 1;

	}

	return sum;

}

export {
	createModel,
	createLatentMipGrids,
	createLatentGrid,
	sampleLatents,
	forwardDecoderInput,
	buildDecoderInput,
	forwardIBLInput,
	buildIBLInput,
	buildIndirectProbeInput,
	unpackIBLOutput,
	getSampleWeightSum,
	dot,
	cross,
	normalize,
	wrapIndex
};

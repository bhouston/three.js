import {
	LEVELS,
	BASE_RESOLUTION,
	TARGET_RESOLUTION,
	CHANNELS_PER_LEVEL,
	IBL_OUTPUT_SIZE,
	INDIRECT_OUTPUT_SIZE,
	computeLatentChannels,
	computeDecoderInputSize,
	computeIblInputSize,
	computeIndirectInputSize
} from './NeuralAppearanceFormat.js';
import {
	createMLP,
	forwardMLP
} from '../neural/NeuralMLP.js';
import { computeGridLevels, createLatentGrid } from '../neural/NeuralGridModel.js';
import { dot, cross, normalize } from '../neural/NeuralVectorMath.js';

function createModel( options, random ) {

	// `levels` is computed first (and every MLP head below is sized from
	// *this* model's actual levels, via the computeXXX helpers) - not from
	// NeuralAppearanceFormat.js's fixed LATENT_CHANNELS/etc. constants - see
	// that file's doc comment on these helpers for why.
	const levels = options.levels || LEVELS;
	const latentChannels = computeLatentChannels( levels );
	const decoderInputSize = computeDecoderInputSize( levels );
	const iblInputSize = computeIblInputSize( levels );
	const indirectInputSize = computeIndirectInputSize( levels );

	const decoder = createMLP( decoderInputSize, [ options.hiddenSize, options.hiddenSize ], 3, random, 'relu', 'linear' );
	const iblHiddenSize = options.iblHiddenSize || Math.min( Math.max( options.hiddenSize || 32, 16 ), 32 );
	const iblHead = createMLP( iblInputSize, [ iblHiddenSize ], IBL_OUTPUT_SIZE, random, 'relu', 'linear' );
	const indirectRadianceHead = createMLP( indirectInputSize, [ iblHiddenSize ], INDIRECT_OUTPUT_SIZE, random, 'relu', 'linear' );
	const indirectIrradianceHead = createMLP( indirectInputSize, [ iblHiddenSize ], INDIRECT_OUTPUT_SIZE, random, 'relu', 'linear' );
	const emissionHead = options.outputFeatures && options.outputFeatures.emission ?
		createMLP( latentChannels, [], 3, random, 'relu', 'linear' ) :
		null;
	const opacityHead = options.outputFeatures && options.outputFeatures.opacity ?
		createMLP( latentChannels, [], 1, random, 'relu', 'linear' ) :
		null;
	const rotationWeights = new Array( latentChannels * 12 ).fill( 0 );

	const baseResolution = options.baseResolution || BASE_RESOLUTION;
	const targetResolution = options.targetResolution || TARGET_RESOLUTION;
	const resolutions = computeGridLevels( baseResolution, targetResolution, levels );
	const latentGrids = resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, CHANNELS_PER_LEVEL, random ) );

	const model = {
		decoder,
		iblHead,
		indirectRadianceHead,
		indirectIrradianceHead,
		emissionHead,
		opacityHead,
		rotationWeights,
		levels,
		baseResolution,
		targetResolution,
		resolutions,
		latentGrids
	};

	decorrelateLinearRgbHead( model );

	return model;

}

function decorrelateLinearRgbHead( model ) {

	const layer = model.decoder.layers[ model.decoder.layers.length - 1 ];
	if ( layer.outputSize !== 3 || model.latentGrids === undefined ) return;

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

		const latents = sampleLatents( model.latentGrids, probe.uv ).output;
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

/**
 * Bilinear-samples every level of a multiresolution latent grid pyramid
 * (same encoding as neural-texture / neural-material - see
 * NeuralGridModel.js) and concatenates their features into a single
 * `LATENT_CHANNELS`-wide latent vector.
 */
function sampleLatents( grids, uv ) {

	const totalChannels = grids.reduce( ( sum, grid ) => sum + grid.channels, 0 );
	const output = new Array( totalChannels ).fill( 0 );
	const levelTaps = [];
	let channelOffset = 0;

	for ( const grid of grids ) {

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

		for ( const tap of taps ) {

			const offset = ( tap.y * grid.width + tap.x ) * grid.channels;

			for ( let channel = 0; channel < grid.channels; channel ++ ) {

				output[ channelOffset + channel ] += grid.data[ offset + channel ] * tap.weight;

			}

		}

		levelTaps.push( { grid, taps, channelOffset } );
		channelOffset += grid.channels;

	}

	return { output, levelTaps };

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
	// Expected width is *this call's* latents.length + 6 (see
	// computeIblInputSize in NeuralAppearanceFormat.js), not the fixed
	// IBL_INPUT_SIZE constant - that constant is only correct when
	// latents.length === LATENT_CHANNELS (i.e. levels === LEVELS).
	const expectedSize = latents.length + 6;

	if ( input.length !== expectedSize ) {

		throw new Error( `THREE.NeuralAppearanceModel: IBL input has ${ input.length } values, expected ${ expectedSize }.` );

	}

	return input;

}

function buildIndirectProbeInput( latents, wo, probe ) {

	const input = latents.slice();
	const incoming = probe || [ 1, 1, 1 ];

	input.push( wo[ 0 ], wo[ 1 ], wo[ 2 ] );
	input.push( incoming[ 0 ], incoming[ 1 ], incoming[ 2 ] );

	// See buildIBLInput's comment above - derived from this call's actual
	// latents.length, not a fixed constant.
	const expectedSize = latents.length + 6;

	if ( input.length !== expectedSize ) {

		throw new Error( `THREE.NeuralAppearanceModel: Indirect probe input has ${ input.length } values, expected ${ expectedSize }.` );

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
	// Row stride is this call's actual latents.length, not the fixed
	// LATENT_CHANNELS constant - `weights` (rotationWeights) is sized
	// `latents.length * ROTATION_OUTPUT_SIZE` by createModel for whatever
	// `levels` that model was actually built with (see createModel's own
	// comment), which only equals LATENT_CHANNELS when levels === LEVELS.
	const stride = latents.length;

	for ( let i = 0; i < latents.length; i ++ ) {

		value += weights[ outputIndex * stride + i ] * latents[ i ];

	}

	return value;

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
	sampleLatents,
	forwardDecoderInput,
	buildDecoderInput,
	forwardIBLInput,
	buildIBLInput,
	buildIndirectProbeInput,
	unpackIBLOutput,
	getSampleWeightSum,
	wrapIndex
};

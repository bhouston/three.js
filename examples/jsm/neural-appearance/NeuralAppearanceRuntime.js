import { CHANNELS_PER_LEVEL } from './NeuralAppearanceFormat.js';
import { sigmoid } from '../neural/NeuralMLP.js';
import {
	buildDecoderInput,
	buildIBLInput,
	buildIndirectProbeInput,
	unpackIBLOutput,
	normalize,
	wrapIndex
} from './NeuralAppearanceModel.js';

const DEFAULT_IBL_INTEGRATION_SAMPLES = 64;

function evaluateNeuralAppearanceJson( json, reference ) {

	return evaluateNeuralAppearanceOutputs( json, reference ).brdf;

}

function evaluateNeuralAppearanceOutputs( json, reference ) {

	const wi = normalize( reference.wi );
	const wo = normalize( reference.wo );
	const brdf = json.outputs.brdf;

	const latents = sampleRuntimeLatents( json, reference.uv || [ 0.5, 0.5 ] );
	const input = buildDecoderInput( latents, brdf.rotation.weights, wi, wo );
	const result = {
		brdf: evaluateDecoderLayers( brdf.layers, input, brdf.outputActivation ),
		ibl: evaluateIBLHead( json, latents, wo )
	};

	if ( json.outputs.indirectRadiance || json.outputs.indirectIrradiance ) {

		const indirect = evaluateIndirectHeads( json, latents, wo, reference );
		result.indirectRadiance = indirect.indirectRadiance;
		result.indirectIrradiance = indirect.indirectIrradiance;
		result.indirect = indirect.indirect;

	}

	if ( json.outputs.emission ) {

		result.emission = evaluateDecoderLayers( json.outputs.emission.layers, latents, json.outputs.emission.outputActivation );

	}

	if ( json.outputs.opacity ) {

		result.opacity = evaluateDecoderLayers( json.outputs.opacity.layers, latents, json.outputs.opacity.outputActivation )[ 0 ];

	}

	return result;

}

function evaluateIndirectHeads( json, latents, wo, reference ) {

	const incoming = ( reference && ( reference.iblIncoming || reference.prefilteredSpecular ) ) || [ 1, 1, 1 ];
	const irradiance = ( reference && reference.iblIrradiance ) || [ 1, 1, 1 ];
	const indirectRadiance = json.outputs.indirectRadiance ?
		evaluateDecoderLayers( json.outputs.indirectRadiance.layers, buildIndirectProbeInput( latents, wo, incoming ), json.outputs.indirectRadiance.outputActivation ) :
		[ 0, 0, 0 ];
	const indirectIrradiance = json.outputs.indirectIrradiance ?
		evaluateDecoderLayers( json.outputs.indirectIrradiance.layers, buildIndirectProbeInput( latents, wo, irradiance ), json.outputs.indirectIrradiance.outputActivation ) :
		[ 0, 0, 0 ];

	return {
		indirectRadiance,
		indirectIrradiance,
		indirect: [
			indirectRadiance[ 0 ] + indirectIrradiance[ 0 ],
			indirectRadiance[ 1 ] + indirectIrradiance[ 1 ],
			indirectRadiance[ 2 ] + indirectIrradiance[ 2 ]
		]
	};

}

function evaluateIBLHead( json, latents, wo ) {

	const ibl = json.outputs.ibl;
	const input = buildIBLInput( latents, json.outputs.brdf.rotation.weights, wo );

	return unpackIBLOutput( evaluateDecoderLayers( ibl.layers, input, { type: 'raw' } ) );

}

function mixArray( a, b, amount ) {

	return a.map( ( value, index ) => value * ( 1 - amount ) + b[ index ] * amount );

}

function evaluateNeuralIBLWhiteFurnace( json, reference ) {

	return evaluateNeuralPrefilteredIBL( json, {
		...reference,
		iblIncoming: [ 1, 1, 1 ],
		iblIrradiance: [ 1, 1, 1 ],
		prefilteredSpecular: [ 1, 1, 1 ]
	} );

}

function evaluateNeuralPrefilteredIBL( json, reference ) {

	const outputs = evaluateNeuralAppearanceOutputs( json, reference );
	if ( outputs.indirect ) return outputs.indirect;

	return [ 0, 0, 0 ];

}

function integrateNeuralBRDFWhiteFurnace( json, reference, sampleCount = DEFAULT_IBL_INTEGRATION_SAMPLES ) {

	const total = [ 0, 0, 0 ];
	const wo = normalize( reference.wo || [ 0, 0, 1 ] );

	for ( let i = 0; i < sampleCount; i ++ ) {

		const wi = sampleHemisphereCosineHammersley( i, sampleCount );
		const brdf = evaluateNeuralAppearanceJson( json, { ...reference, wi, wo } );

		for ( let channel = 0; channel < 3; channel ++ ) {

			total[ channel ] += brdf[ channel ] / sampleCount * Math.PI;

		}

	}

	return total;

}

function sampleHemisphereCosineHammersley( index, count ) {

	const u = ( index + 0.5 ) / Math.max( 1, count );
	const v = radicalInverseVdc( index );
	const r = Math.sqrt( u );
	const phi = 2 * Math.PI * v;

	return [
		r * Math.cos( phi ),
		r * Math.sin( phi ),
		Math.sqrt( Math.max( 0, 1 - u ) )
	];

}

function radicalInverseVdc( bits ) {

	bits = ( bits << 16 ) | ( bits >>> 16 );
	bits = ( ( bits & 0x55555555 ) << 1 ) | ( ( bits & 0xAAAAAAAA ) >>> 1 );
	bits = ( ( bits & 0x33333333 ) << 2 ) | ( ( bits & 0xCCCCCCCC ) >>> 2 );
	bits = ( ( bits & 0x0F0F0F0F ) << 4 ) | ( ( bits & 0xF0F0F0F0 ) >>> 4 );
	bits = ( ( bits & 0x00FF00FF ) << 8 ) | ( ( bits & 0xFF00FF00 ) >>> 8 );

	return ( bits >>> 0 ) * 2.3283064365386963e-10;

}

/**
 * Bilinear-samples every level of the multiresolution latent grid (same
 * encoding as neural-texture / neural-material - see NeuralGridModel.js) and
 * concatenates their channels into a single `levels.length *
 * CHANNELS_PER_LEVEL`-wide latent vector. No mip/LOD selection - every level
 * always contributes.
 *
 * Sized from `json.latents.levels.length` - the manifest's *actual* level
 * count - not NeuralAppearanceFormat.js's fixed `LATENT_CHANNELS` constant,
 * which is only correct for a manifest exported with the default `levels`
 * (4). A fixed-size allocation here would leave phantom trailing zero
 * "channels" for a manifest with fewer levels (corrupting every downstream
 * consumer that derives its own expected input width from `latents.length`,
 * as buildDecoderInput/buildIBLInput/buildIndirectProbeInput in
 * NeuralAppearanceModel.js now do), and for a manifest with more levels than
 * the default, would silently rely on the array auto-growing rather than
 * being sized correctly up front. See NeuralAppearanceFormat.js's doc
 * comment on the computeXXX helpers for the full story.
 */
function sampleRuntimeLatents( json, uv ) {

	const levels = json.latents.levels;
	const latents = new Array( levels.length * CHANNELS_PER_LEVEL ).fill( 0 );
	let channelOffset = 0;

	for ( const level of levels ) {

		const x = uv[ 0 ] * level.width - 0.5;
		const y = uv[ 1 ] * level.height - 0.5;
		const x0 = Math.floor( x );
		const y0 = Math.floor( y );
		const tx = x - x0;
		const ty = y - y0;
		const repeat = level.wrap !== 'clamp';
		const taps = [
			{ x: x0, y: y0, weight: ( 1 - tx ) * ( 1 - ty ) },
			{ x: x0 + 1, y: y0, weight: tx * ( 1 - ty ) },
			{ x: x0, y: y0 + 1, weight: ( 1 - tx ) * ty },
			{ x: x0 + 1, y: y0 + 1, weight: tx * ty }
		];

		for ( const tap of taps ) {

			const tapX = repeat ? wrapIndex( tap.x, level.width ) : Math.min( Math.max( tap.x, 0 ), level.width - 1 );
			const tapY = repeat ? wrapIndex( tap.y, level.height ) : Math.min( Math.max( tap.y, 0 ), level.height - 1 );
			const offset = ( tapY * level.width + tapX ) * CHANNELS_PER_LEVEL;

			for ( let channel = 0; channel < CHANNELS_PER_LEVEL; channel ++ ) {

				latents[ channelOffset + channel ] += level.data[ offset + channel ] * tap.weight;

			}

		}

		channelOffset += CHANNELS_PER_LEVEL;

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

	if ( outputActivation.type === 'sigmoid' ) {

		return values.map( sigmoid );

	}

	if ( outputActivation.type === 'raw' ) {

		return values;

	}

	return values.map( ( value ) => Math.max( value, 0 ) );

}

export {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	evaluateIBLHead,
	evaluateNeuralPrefilteredIBL,
	evaluateNeuralIBLWhiteFurnace,
	integrateNeuralBRDFWhiteFurnace,
	sampleRuntimeLatents,
	evaluateDecoderLayers,
	mixArray
};

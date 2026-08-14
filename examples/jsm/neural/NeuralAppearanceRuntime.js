import { DataUtils } from 'three';
import { LATENT_CHANNELS } from './NeuralAppearanceFormat.js';
import { sigmoid } from './NeuralAppearanceMLP.js';
import { buildDecoderInput, normalize, wrapIndex } from './NeuralAppearanceModel.js';

function evaluateNeuralAppearanceJson( json, reference ) {

	return evaluateNeuralAppearanceOutputs( json, reference ).brdf;

}

function evaluateNeuralAppearanceOutputs( json, reference ) {

	const mip = selectRuntimeMipLevel( json, reference );
	const latents = sampleRuntimeLatents( json, reference.uv || [ 0.5, 0.5 ], mip );
	const wi = normalize( reference.wi );
	const wo = normalize( reference.wo );
	const brdf = json.outputs.brdf;
	const input = buildDecoderInput( latents, brdf.rotation.weights, wi, wo );
	const result = {
		brdf: evaluateDecoderLayers( brdf.layers, input, brdf.outputActivation )
	};

	if ( json.outputs.emission ) {

		result.emission = evaluateDecoderLayers( json.outputs.emission.layers, latents, json.outputs.emission.outputActivation );

	}

	if ( json.outputs.opacity ) {

		result.opacity = evaluateDecoderLayers( json.outputs.opacity.layers, latents, json.outputs.opacity.outputActivation )[ 0 ];

	}

	return result;

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

	if ( outputActivation.type === 'sigmoid' ) {

		return values.map( sigmoid );

	}

	return values.map( ( value ) => Math.max( 0, value ) );

}

export {
	evaluateNeuralAppearanceJson,
	evaluateNeuralAppearanceOutputs,
	selectRuntimeMipLevel,
	sampleRuntimeLatents,
	evaluateDecoderLayers
};

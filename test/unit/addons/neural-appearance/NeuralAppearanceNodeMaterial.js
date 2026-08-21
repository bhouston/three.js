import { describe, test, expect } from 'vitest';
import { NeuralAppearanceLoader } from '../../../../examples/jsm/loaders/NeuralAppearanceLoader.js';
import { NeuralAppearanceNodeMaterial } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceNodeMaterial.js';
import { evaluateNeuralDebugShading } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTSL.js';
import { encodeUint8Base64, encodeFloat16Base64, encodeMLPLayersBase64 } from '../../../../examples/jsm/neural/NeuralBinaryCodec.js';

// Builds a compact (uint8-quantized latents, float16-packed MLP weights)
// `.neuralAppearance` manifest by hand, mirroring exactly what
// NeuralAppearanceManifest.js's `compactAppearanceJson` produces from a real
// trained model - see that file for the authoritative shape.

function encodeLevel( fill ) {

	const data = [ fill, 0, 0, 0 ];
	const min = Math.min( 0, fill );
	const max = Math.max( 0, fill );

	return {
		width: 1,
		height: 1,
		channels: 4,
		wrap: 'repeat',
		dtype: 'uint8',
		min,
		max,
		dataBase64: encodeUint8Base64( Float32Array.from( data ), min, max )
	};

}

function encodeHead( inputSize, layers, outputActivation, extra = {} ) {

	return {
		inputSize,
		mlp: encodeMLPLayersBase64( layers ),
		outputActivation,
		...extra
	};

}

function encodeRotation( inputSize, outputSize, weights ) {

	return {
		inputSize,
		outputSize,
		dtype: 'float16',
		dataBase64: encodeFloat16Base64( Float32Array.from( weights ) )
	};

}

function createManifest( fill = 0 ) {

	return {
		format: 'three-neural-appearance',
		version: 9,
		name: 'unit test material',
		latents: {
			levels: [ encodeLevel( fill ), encodeLevel( 0 ), encodeLevel( 0 ), encodeLevel( 0 ) ],
			channelsPerLevel: 4,
			wrap: 'repeat'
		},
		outputs: {
			brdf: {
				inputSize: 28,
				rotation: encodeRotation( 16, 12, new Array( 192 ).fill( fill ) ),
				...encodeHead( 28, [
					{ inputSize: 28, outputSize: 3, activation: 'linear', biases: [ fill, 0, 0 ], weights: new Array( 84 ).fill( fill ) }
				], { type: 'linear' } )
			},
			ibl: encodeHead( 22, [
				{ inputSize: 22, outputSize: 4, activation: 'linear', biases: [ 0, 0, 1, 0 ], weights: new Array( 22 * 4 ).fill( fill ) }
			], { type: 'linear' } ),
			indirectRadiance: encodeHead( 22, [
				{ inputSize: 22, outputSize: 3, activation: 'linear', biases: [ 0, 0, 0 ], weights: new Array( 22 * 3 ).fill( fill ) }
			], { type: 'linear' } ),
			indirectIrradiance: encodeHead( 22, [
				{ inputSize: 22, outputSize: 3, activation: 'linear', biases: [ 0, 0, 0 ], weights: new Array( 22 * 3 ).fill( fill ) }
			], { type: 'linear' } )
		}
	};

}

describe( 'Addons', () => {

	describe( 'Neural', () => {

		describe( 'NeuralAppearanceNodeMaterial', () => {

			test( 'updates matching weights without rebuilding', () => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const textures = material.neuralAppearanceData.latentTextures;
				const weightVector = material._outputUniforms.brdf.parameters.array[ material._outputUniforms.brdf.layers[ 0 ].weightsOffset ];
				const next = loader.parse( createManifest( 0.25 ) );

				expect( material.updateFromData( next ) ).toBe( true );
				expect( material.neuralAppearanceData.latentTextures ).toBe( textures );
				expect( material._outputUniforms.brdf.parameters.array[ material._outputUniforms.brdf.layers[ 0 ].weightsOffset ] ).toBe( weightVector );
				expect( weightVector.x ).not.toBe( 0 );

			} );

			test( 'rejects a different decoder layout', () => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const nextManifest = createManifest( 0 );
				nextManifest.outputs.brdf = {
					inputSize: 28,
					rotation: encodeRotation( 16, 12, new Array( 192 ).fill( 0 ) ),
					...encodeHead( 28, [
						{ inputSize: 28, outputSize: 8, activation: 'relu', biases: new Array( 8 ).fill( 0 ), weights: new Array( 224 ).fill( 0 ) },
						{ inputSize: 8, outputSize: 3, activation: 'linear', biases: new Array( 3 ).fill( 0 ), weights: new Array( 24 ).fill( 0 ) }
					], { type: 'linear' } )
				};

				expect( material.updateFromData( loader.parse( nextManifest ) ) ).toBe( false );

			} );

			test( 'configures mask and blend opacity runtime state', () => {

				const loader = new NeuralAppearanceLoader();

				const maskManifest = createManifest( 0 );
				maskManifest.outputs.opacity = encodeHead( 16, [
					{ inputSize: 16, outputSize: 1, activation: 'linear', biases: [ 0 ], weights: [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] }
				], { type: 'sigmoid' }, { mode: 'mask', alphaCutoff: 0.5 } );

				const maskMaterial = new NeuralAppearanceNodeMaterial( loader.parse( maskManifest ) );
				expect( maskMaterial.transparent ).toBe( false );
				expect( maskMaterial.alphaTest ).toBe( 0.5 );
				expect( maskMaterial.alphaTestNode ).toBeTruthy();
				expect( maskMaterial.opacityNode ).toBeTruthy();
				expect( maskMaterial.maskNode ).toBeTruthy();

				const blendManifest = createManifest( 0 );
				blendManifest.outputs.opacity = encodeHead( 16, [
					{ inputSize: 16, outputSize: 1, activation: 'linear', biases: [ 0 ], weights: [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] }
				], { type: 'sigmoid' }, { mode: 'blend' } );

				const blendMaterial = new NeuralAppearanceNodeMaterial( loader.parse( blendManifest ) );
				expect( blendMaterial.transparent ).toBe( true );
				expect( blendMaterial.depthWrite ).toBe( false );
				expect( blendMaterial.opacityNode ).toBeTruthy();
				expect( blendMaterial.alphaTestNode ).toBe( null );
				expect( blendMaterial.maskNode ).toBe( null );

				expect( maskMaterial.updateFromData( loader.parse( blendManifest ) ) ).toBe( false );

			} );

			test( 'builds debug shading graphs for decoder-frame visualization', () => {

				const loader = new NeuralAppearanceLoader();
				const material = new NeuralAppearanceNodeMaterial( loader.parse( createManifest( 0 ) ) );
				const debug = evaluateNeuralDebugShading( material );

				expect( material.debugView ).toBe( 'shaded' );
				expect( debug.viewNormal ).toBeTruthy();
				expect( debug.viewReflect ).toBeTruthy();
				expect( debug.viewIrradiance ).toBeTruthy();
				expect( debug.roughness ).toBeTruthy();

				material.debugView = 'roughness';
				expect( material.debugView ).toBe( 'roughness' );
				material.debugView = 'ibl';
				expect( material.debugView ).toBe( 'ibl' );

			} );

		} );

	} );

} );

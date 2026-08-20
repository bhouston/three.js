import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { float, vec3, vec4 } from 'three/tsl';
import {
	createEvaluateNeuralBRDFFn,
	evaluateNeuralEmission,
	evaluateNeuralOpacity,
	createOutputUniforms
} from '../../../../examples/jsm/neural-appearance/NeuralAppearanceTSL.js';
import { buildDecoderInput } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceModel.js';
import { normalize } from '../../../../examples/jsm/neural/NeuralVectorMath.js';
import { evaluateDecoderLayers } from '../../../../examples/jsm/neural-appearance/NeuralAppearanceRuntime.js';
import { createRandom } from '../../../../examples/jsm/neural/NeuralTrainingUtils.js';
import { withTestRenderer, evalFloats, evalScalar } from '../helpers/webgpuEval.js';

// NeuralAppearanceTSL.js is the GPU-side twin of NeuralAppearanceRuntime.js's
// CPU reference decoder - both consume the exact same manifest shape (layers
// of {weights, biases, inputSize, outputSize, activation} plus a rotation
// sub-layer and an outputActivation), and are otherwise two entirely
// independent implementations (hand-written JS array math on the CPU side vs
// a packed-vec4 TSL node graph evaluated by the WebGPU compute pipeline on
// the GPU side). This file builds one small hand-crafted manifest, feeds it
// through both, and asserts they agree - the same comparison
// `createReferenceEvaluations` in NeuralAppearanceManifest.js already
// produces at export time as JSON for a human to eyeball, made into a real
// assertion. A regression in either implementation that made it disagree
// with the other would fail here.

const LEVELS = 4;
const LATENT_CHANNELS = LEVELS * 4; // `LEVELS` levels x 4 channels/level

function buildLayer( random, inputSize, outputSize, activation ) {

	const weights = Array.from( { length: inputSize * outputSize }, () => ( random() - 0.5 ) );
	const biases = Array.from( { length: outputSize }, () => ( random() - 0.5 ) * 0.2 );

	return { inputSize, outputSize, activation, weights, biases };

}

function buildManifestOutputs( random, latentChannels = LATENT_CHANNELS ) {

	const decoderInputSize = latentChannels + 12;
	const iblInputSize = latentChannels + 6;

	const brdf = {
		inputSize: decoderInputSize,
		rotation: {
			inputSize: latentChannels,
			outputSize: 12,
			weights: Array.from( { length: latentChannels * 12 }, () => ( random() - 0.5 ) )
		},
		layers: [
			buildLayer( random, decoderInputSize, 8, 'relu' ),
			buildLayer( random, 8, 3, 'linear' )
		],
		outputActivation: { type: 'scaledSigmoid', scale: 1 }
	};

	const ibl = {
		inputSize: iblInputSize,
		layers: [
			buildLayer( random, iblInputSize, 8, 'relu' ),
			buildLayer( random, 8, 4, 'linear' )
		],
		outputActivation: { type: 'raw' }
	};

	const emission = {
		inputSize: latentChannels,
		layers: [ buildLayer( random, latentChannels, 3, 'linear' ) ],
		outputActivation: { type: 'exp', offset: 0 }
	};

	const opacity = {
		inputSize: latentChannels,
		layers: [ buildLayer( random, latentChannels, 1, 'linear' ) ],
		outputActivation: { type: 'sigmoid' }
	};

	return { brdf, ibl, emission, opacity };

}

let nextMaterialId = 1;

function buildMaterialStub( outputs, levels = LEVELS ) {

	return {
		id: nextMaterialId ++, // unique per stub - createEvaluateNeuralBRDFFn bakes this into its generated function name
		neuralAppearanceData: { outputs, levels },
		_outputUniforms: createOutputUniforms( outputs )
	};

}

async function evalVec3( renderer, buildNode ) {

	return evalFloats( renderer, 3, ( out ) => {

		const result = buildNode();
		out.element( 0 ).assign( result.x );
		out.element( 1 ).assign( result.y );
		out.element( 2 ).assign( result.z );

	} );

}

function vec4FromLatents( latents, offset ) {

	return vec4( latents[ offset ], latents[ offset + 1 ], latents[ offset + 2 ], latents[ offset + 3 ] );

}

// Builds the `{ latent0: vec4(...), latent1: vec4(...), ... }` args
// createEvaluateNeuralBRDFFn's generated Fn expects, for however many levels
// `latents` actually holds (`latents.length / 4`) - mirrors
// NeuralAppearanceTSL.js's own `levelNames`.
function latentArgsFromArray( latents ) {

	const args = {};

	for ( let level = 0; level * 4 < latents.length; level ++ ) {

		args[ `latent${ level }` ] = vec4FromLatents( latents, level * 4 );

	}

	return args;

}

describe( 'Addons > NeuralAppearance > NeuralAppearanceTSL (real WebGPU, cross-checked against NeuralAppearanceRuntime CPU oracle)', () => {

	const getRenderer = withTestRenderer( { beforeAll, afterAll } );

	// One fixed hand-crafted "manifest" (small weights/biases/latents, same
	// shape createNeuralAppearanceManifest produces) shared by every test in
	// this file, so the CPU and GPU sides always consume identical input data.
	const random = createRandom( 20260820 );
	const outputs = buildManifestOutputs( random );
	const material = buildMaterialStub( outputs );
	const latents = Array.from( { length: LATENT_CHANNELS }, () => ( random() - 0.5 ) );

	const directionSets = [
		{ wi: [ 0, 0, 1 ], wo: normalize( [ 0.4, 0.2, 0.894 ] ) },
		{ wi: normalize( [ 0.5, 0.1, 0.86 ] ), wo: normalize( [ - 0.3, 0.4, 0.86 ] ) },
		{ wi: normalize( [ - 0.4, 0.3, 0.866 ] ), wo: [ 0, 0, 1 ] },
		{ wi: normalize( [ 0.1, - 0.6, 0.79 ] ), wo: normalize( [ 0.7, 0.1, 0.71 ] ) }
	];

	describe( 'BRDF head (createEvaluateNeuralBRDFFn) vs NeuralAppearanceRuntime.evaluateDecoderLayers', () => {

		for ( let i = 0; i < directionSets.length; i ++ ) {

			const { wi, wo } = directionSets[ i ];

			it( `direction set ${i}: GPU decoder output matches the CPU reference decoder`, async () => {

				// Independent CPU oracle: NeuralAppearanceModel.buildDecoderInput +
				// NeuralAppearanceRuntime.evaluateDecoderLayers, hand-written array
				// math with no TSL/GPU involvement.
				const cpuInput = buildDecoderInput( latents, outputs.brdf.rotation.weights, wi, wo );
				const cpuDecoded = evaluateDecoderLayers( outputs.brdf.layers, cpuInput, outputs.brdf.outputActivation );
				const expected = cpuDecoded.map( ( value ) => value * Math.max( wi[ 2 ], 0 ) );

				const fn = createEvaluateNeuralBRDFFn( material );
				const gpuResult = await evalVec3( getRenderer(), () => fn( {
					wi: vec3( wi[ 0 ], wi[ 1 ], wi[ 2 ] ),
					wo: vec3( wo[ 0 ], wo[ 1 ], wo[ 2 ] ),
					...latentArgsFromArray( latents )
				} ) );

				for ( let channel = 0; channel < 3; channel ++ ) {

					expect( gpuResult[ channel ] ).toBeCloseTo( expected[ channel ], 3 );

				}

			} );

		}

	} );

	it( 'BRDF head: zero weights/biases collapse to the closed-form constant scaledSigmoid(0) * cos(theta_i) (hand-derived, not read from source)', async () => {

		// An edge case whose expected value comes from closed-form math, not
		// from re-reading the implementation: with every weight and bias zero,
		// every hidden pre-activation is exactly 0 regardless of latents/wi/wo,
		// so the decoder output is scaledSigmoid(0) = scale / (1 + e^0) = scale/2
		// on every channel, i.e. 0.5 here (scale = 1), times max(wi.z, 0).
		const zeroOutputs = {
			brdf: {
				inputSize: LATENT_CHANNELS + 12,
				rotation: { inputSize: LATENT_CHANNELS, outputSize: 12, weights: new Array( LATENT_CHANNELS * 12 ).fill( 0 ) },
				layers: [
					{ inputSize: LATENT_CHANNELS + 12, outputSize: 8, activation: 'relu', weights: new Array( ( LATENT_CHANNELS + 12 ) * 8 ).fill( 0 ), biases: new Array( 8 ).fill( 0 ) },
					{ inputSize: 8, outputSize: 3, activation: 'linear', weights: new Array( 8 * 3 ).fill( 0 ), biases: new Array( 3 ).fill( 0 ) }
				],
				outputActivation: { type: 'scaledSigmoid', scale: 1 }
			},
			ibl: outputs.ibl // createOutputUniforms requires an ibl head; unused by this test
		};
		const zeroMaterial = buildMaterialStub( zeroOutputs );
		const zeroLatents = new Array( LATENT_CHANNELS ).fill( 0 );
		const wi = normalize( [ 0.3, 0.2, 0.9 ] );
		const wo = [ 0, 0, 1 ];

		const fn = createEvaluateNeuralBRDFFn( zeroMaterial );
		const gpuResult = await evalVec3( getRenderer(), () => fn( {
			wi: vec3( wi[ 0 ], wi[ 1 ], wi[ 2 ] ),
			wo: vec3( wo[ 0 ], wo[ 1 ], wo[ 2 ] ),
			...latentArgsFromArray( zeroLatents )
		} ) );

		const expected = 0.5 * Math.max( wi[ 2 ], 0 );

		for ( let channel = 0; channel < 3; channel ++ ) {

			expect( gpuResult[ channel ] ).toBeCloseTo( expected, 5 );

		}

	} );

	describe( 'BRDF head with a non-default level count (createEvaluateNeuralBRDFFn generates its latentN inputs from the model)', () => {

		// NeuralAppearanceTSL.js used to hardcode exactly 4 named latent-texel
		// inputs (latent0..latent3); createEvaluateNeuralBRDFFn now generates
		// `material.neuralAppearanceData.levels` of them instead (see
		// levelNames). This exercises that generalization directly with a
		// level count other than the "everything happens to be 4" default
		// used everywhere else in this file, so a regression back to a fixed
		// count would fail here even though the rest of the file wouldn't
		// notice.
		const altRandom = createRandom( 20260821 );
		const altLevels = 2;
		const altLatentChannels = altLevels * 4;
		const altOutputs = buildManifestOutputs( altRandom, altLatentChannels );
		const altMaterial = buildMaterialStub( altOutputs, altLevels );
		const altLatents = Array.from( { length: altLatentChannels }, () => ( altRandom() - 0.5 ) );

		it( 'GPU decoder output matches the CPU reference decoder for a 2-level model', async () => {

			const { wi, wo } = directionSets[ 0 ];

			const cpuInput = buildDecoderInput( altLatents, altOutputs.brdf.rotation.weights, wi, wo );
			const cpuDecoded = evaluateDecoderLayers( altOutputs.brdf.layers, cpuInput, altOutputs.brdf.outputActivation );
			const expected = cpuDecoded.map( ( value ) => value * Math.max( wi[ 2 ], 0 ) );

			const fn = createEvaluateNeuralBRDFFn( altMaterial );
			const gpuResult = await evalVec3( getRenderer(), () => fn( {
				wi: vec3( wi[ 0 ], wi[ 1 ], wi[ 2 ] ),
				wo: vec3( wo[ 0 ], wo[ 1 ], wo[ 2 ] ),
				...latentArgsFromArray( altLatents )
			} ) );

			for ( let channel = 0; channel < 3; channel ++ ) {

				expect( gpuResult[ channel ] ).toBeCloseTo( expected[ channel ], 3 );

			}

		} );

	} );

	// The IBL head's evaluation functions (evaluateLearnedIBLQuery,
	// evaluateLearnedIBLQueryForTexels, buildDecoderFrames, evaluateNeuralIBL)
	// are not exported from NeuralAppearanceTSL.js, and evaluateNeuralIBL - the
	// one that is exported - additionally requires a real env-map node plus
	// TSL.materialEnvIntensity (a live NodeMaterial context), which isn't cheap
	// to stand up in a bare compute-kernel harness. Not covered here; a
	// dedicated test would need either those internals exported or a full
	// NeuralAppearanceNodeMaterial + WebGPURenderer render pass.

	describe( 'Emission head (evaluateNeuralEmission) vs NeuralAppearanceRuntime.evaluateDecoderLayers', () => {

		it( 'GPU emission output (exp activation) matches the CPU reference', async () => {

			const expected = evaluateDecoderLayers( outputs.emission.layers, latents, outputs.emission.outputActivation );

			const context = { latents: latents.map( ( value ) => float( value ) ) };
			const gpuResult = await evalVec3( getRenderer(), () => evaluateNeuralEmission( material, context ) );

			for ( let channel = 0; channel < 3; channel ++ ) {

				expect( gpuResult[ channel ] ).toBeCloseTo( expected[ channel ], 3 );

			}

		} );

	} );

	describe( 'Opacity head (evaluateNeuralOpacity) vs NeuralAppearanceRuntime.evaluateDecoderLayers', () => {

		it( 'GPU opacity output (sigmoid activation) matches the CPU reference', async () => {

			const expected = evaluateDecoderLayers( outputs.opacity.layers, latents, outputs.opacity.outputActivation )[ 0 ];

			const context = { latents: latents.map( ( value ) => float( value ) ) };
			const gpuResult = await evalScalar( getRenderer(), () => evaluateNeuralOpacity( material, context ) );

			expect( gpuResult ).toBeCloseTo( expected, 3 );
			// sigmoid output is always in (0, 1) - a sanity bound independent of
			// the exact weights, catches e.g. an accidentally-dropped activation.
			expect( gpuResult ).toBeGreaterThan( 0 );
			expect( gpuResult ).toBeLessThan( 1 );

		} );

		it( 'all-zero latents/weights collapse to the closed-form constant sigmoid(bias) (hand-derived edge case)', async () => {

			const zeroOutputs = {
				brdf: outputs.brdf, // unused by this test, but required by createOutputUniforms
				ibl: outputs.ibl,
				opacity: {
					inputSize: LATENT_CHANNELS,
					layers: [ { inputSize: LATENT_CHANNELS, outputSize: 1, activation: 'linear', weights: new Array( LATENT_CHANNELS ).fill( 0 ), biases: [ 1.25 ] } ],
					outputActivation: { type: 'sigmoid' }
				}
			};
			const zeroMaterial = buildMaterialStub( zeroOutputs );
			const zeroContext = { latents: new Array( LATENT_CHANNELS ).fill( 0 ).map( ( value ) => float( value ) ) };

			const gpuResult = await evalScalar( getRenderer(), () => evaluateNeuralOpacity( zeroMaterial, zeroContext ) );
			const expected = 1 / ( 1 + Math.exp( - 1.25 ) ); // sigmoid(1.25), computed by hand

			expect( gpuResult ).toBeCloseTo( expected, 5 );

		} );

	} );

} );

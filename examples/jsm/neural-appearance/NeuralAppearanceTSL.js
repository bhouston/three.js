import * as THREE from 'three';
import * as TSL from 'three/tsl';
import {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4
} from '../neural/NeuralMLPTSL.js';
import { sigmoidTSL } from '../neural/NeuralOutputActivations.js';

// The multiresolution latent grid can have any number of levels (see
// computeGridLevels/NeuralAppearanceTrainer's `levels` option) - this
// generates that many named `latent0..latentN-1` vec4 texel inputs for a
// given model's TSL BRDF Fn signature (built once per material, in
// createEvaluateNeuralBRDFFn below), rather than hardcoding a fixed count.
// TSL.Fn's signature has to be static per-Fn-instance (its `setLayout` shape
// is baked into the compiled WGSL function), so this can't be a runtime
// loop over a texture array - it's a JS-time loop, generating a differently-
// shaped Fn per model, which is fine since one is only ever built once per
// material (see createNeuralFragmentContext/NeuralAppearanceNodeMaterial's
// constructor).
function levelNames( levelCount ) {

	return Array.from( { length: levelCount }, ( _, i ) => `latent${ i }` );

}

function createEvaluateNeuralBRDFFn( material ) {

	const brdf = material.neuralAppearanceData.outputs.brdf;
	const uniforms = material._outputUniforms.brdf;
	const names = levelNames( material.neuralAppearanceData.levels );

	return TSL.Fn( ( args ) => {

		const latents = latentsFromTexels( names.map( ( name ) => args[ name ] ) );
		const frames = buildDecoderFrames( brdf, uniforms, latents );
		const input = projectDecoderInput( latents, frames, args.wi, args.wo, brdf.inputSize );
		const decoded = toVec3( evaluateMLP( brdf.layers, uniforms, input ) );

		return applyOutputActivation( decoded, brdf.outputActivation ).mul( args.wi.z.max( 0 ) );

	} ).setLayout( {
		name: `evaluateNeuralBRDF_${ material.id }`,
		type: 'vec3',
		inputs: [
			{ name: 'wi', type: 'vec3' },
			{ name: 'wo', type: 'vec3' },
			...names.map( ( name ) => ( { name, type: 'vec4' } ) )
		]
	} );

}

function evaluateNeuralBRDF( material, lightDirection, context, evaluateFn ) {

	const fragment = context || createNeuralFragmentContext( material );
	const fn = evaluateFn || createEvaluateNeuralBRDFFn( material );
	const wi = transformToCanonicalFrame( lightDirection );

	return fn( wi, fragment.viewDirection, ...fragment.texels );

}

// `context`, when given, is a NeuralAppearanceFragmentContext (see
// createNeuralFragmentContext) shared with the BRDF/IBL evaluation for the
// same fragment -- reuses its already-sampled latent code instead of
// re-sampling all 4 grid-level textures again. Falls back to sampling its
// own copy when called standalone (e.g. `context` not yet available).
function evaluateNeuralEmission( material, context = null ) {

	const output = material.neuralAppearanceData.outputs.emission;
	const uniforms = material._outputUniforms.emission;
	const latents = context ? context.latents : fetchLatentCode( material );
	const decoded = toVec3( evaluateMLPViaFn( `evaluateEmissionHead_${ material.id }`, output.layers, uniforms, latents, 3 ) );

	return applyOutputActivation( decoded, output.outputActivation );

}

function evaluateNeuralOpacity( material, context = null ) {

	const output = material.neuralAppearanceData.outputs.opacity;
	const uniforms = material._outputUniforms.opacity;
	const latents = context ? context.latents : fetchLatentCode( material );
	const decoded = evaluateMLPViaFn( `evaluateOpacityHead_${ material.id }`, output.layers, uniforms, latents, 1 )[ 0 ];

	return applyScalarOutputActivation( decoded, output.outputActivation );

}

function evaluateNeuralIBL( material, envNode, context = null, isolate = 'full' ) {

	const fragment = context || createNeuralFragmentContext( material );

	return evaluateNeuralIBLForTexels( material, envNode, fragment.viewDirection, fragment.latents, fragment.frames, isolate );

}

// `latents`/`frames` are the same per-fragment values
// createNeuralFragmentContext already computed for BRDF -- passed in
// directly instead of re-sampling the grid textures or re-running the
// rotation decoder (buildDecoderFrames) here a second time. Not
// exported/called outside this module, so its signature is free to change.
function evaluateNeuralIBLForTexels( material, envNode, wo, latents, frames, isolate = 'full' ) {

	const ibl = material.neuralAppearanceData.outputs.ibl;
	const uniforms = material._outputUniforms.ibl;

	const queryInput = projectIBLInput( latents, frames, wo, ibl.inputSize );
	const query = evaluateMLPViaFn( `evaluateIBLHead_${ material.id }`, ibl.layers, uniforms, queryInput, 4 );
	const queryDirection = TSL.vec3( query[ 0 ], query[ 1 ], query[ 2 ] ).normalize();
	const queryRoughness = sigmoidTSL( query[ 3 ] );

	const incoming = envNode.context( {
		getUV: () => canonicalToWorldDirection( queryDirection ),
		getTextureLevel: () => queryRoughness
	} ).isolate().mul( TSL.materialEnvIntensity );
	const irradiance = envNode.context( {
		getUV: () => canonicalToWorldDirection( TSL.vec3( 0, 0, 1 ) ),
		getTextureLevel: () => TSL.float( 1 )
	} ).isolate().mul( TSL.materialEnvIntensity );
	const radianceHead = material.neuralAppearanceData.outputs.indirectRadiance;
	const irradianceHead = material.neuralAppearanceData.outputs.indirectIrradiance;
	let outgoing = TSL.vec3( 0 );

	if ( isolate !== 'irradiance' && radianceHead ) {

		outgoing = outgoing.add( evaluateIndirectProbeHead( material, radianceHead, material._outputUniforms.indirectRadiance, latents, wo, incoming, 'Radiance' ) );

	}

	if ( isolate !== 'radiance' && irradianceHead ) {

		outgoing = outgoing.add( evaluateIndirectProbeHead( material, irradianceHead, material._outputUniforms.indirectIrradiance, latents, wo, irradiance, 'Irradiance' ) );

	}

	if ( radianceHead || irradianceHead ) return outgoing;

	return isolate === 'irradiance' ? irradiance : incoming;

}

function getCanonicalViewBasis() {

	const frame = TSL.TBNViewMatrix;
	const normal = frame[ 2 ].normalize();

	// The derivative TBN fallback uses a shared scale for both axes. Rebuild an
	// orthonormal basis so runtime directions match the canonical training frame.
	const projectedTangent = frame[ 0 ].sub( normal.mul( frame[ 0 ].dot( normal ) ) );
	const tangentLengthSquared = projectedTangent.dot( projectedTangent );
	const normalizedTangent = projectedTangent.mul( tangentLengthSquared.max( 1e-10 ).inverseSqrt() );
	const fallbackAxis = normal.y.abs().lessThan( 0.999 ).select( TSL.vec3( 0, 1, 0 ), TSL.vec3( 1, 0, 0 ) );
	const fallbackTangent = fallbackAxis.cross( normal ).normalize();
	const tangent = tangentLengthSquared.greaterThan( 1e-10 ).select( normalizedTangent, fallbackTangent );
	const unhandedBitangent = normal.cross( tangent );
	const handedness = unhandedBitangent.dot( frame[ 1 ] ).lessThan( 0 ).select( - 1, 1 );
	const bitangent = unhandedBitangent.mul( handedness );

	return { tangent, bitangent, normal };

}

function canonicalToViewDirection( direction ) {

	const { tangent, bitangent, normal } = getCanonicalViewBasis();

	return tangent.mul( direction.x ).add( bitangent.mul( direction.y ) ).add( normal.mul( direction.z ) ).normalize();

}

function canonicalToWorldDirection( direction ) {

	return canonicalToViewDirection( direction ).transformDirection( TSL.cameraWorldMatrix );

}

function transformToCanonicalFrame( direction ) {

	const { tangent, bitangent, normal } = getCanonicalViewBasis();

	return TSL.vec3(
		direction.dot( tangent ),
		direction.dot( bitangent ),
		direction.dot( normal )
	).normalize();

}

// `texels` is an array (one vec4 node per grid level, in level order - see
// fetchLatentTexels), not a fixed count - flattened to a plain array of
// scalar latent nodes 4 at a time.
function latentsFromTexels( texels ) {

	return texels.flatMap( ( texel ) => [ texel.x, texel.y, texel.z, texel.w ] );

}

function evaluateLearnedCanonicalNormal( material, fragment ) {

	const brdf = material.neuralAppearanceData.outputs.brdf;
	const uniforms = material._outputUniforms.brdf;

	return buildDecoderFrames( brdf, uniforms, latentsFromTexels( fragment.texels ) )[ 0 ].n;

}

function evaluateLearnedIBLQueryForTexels( material, texels, wo ) {

	const latents = latentsFromTexels( texels );
	const ibl = material.neuralAppearanceData.outputs.ibl;
	const frames = buildDecoderFrames( material.neuralAppearanceData.outputs.brdf, material._outputUniforms.brdf, latents );
	const input = projectIBLInput( latents, frames, wo, ibl.inputSize );
	const output = evaluateMLPViaFn( `evaluateIBLHead_${ material.id }`, ibl.layers, material._outputUniforms.ibl, input, 4 );

	return {
		direction: TSL.vec3( output[ 0 ], output[ 1 ], output[ 2 ] ).normalize(),
		roughness: sigmoidTSL( output[ 3 ] )
	};

}

function evaluateLearnedIBLQuery( material, fragment ) {

	return evaluateLearnedIBLQueryForTexels( material, fragment.texels, fragment.viewDirection );

}

/**
 * Decodes learned shading-frame intermediates used by the debug visualizer.
 *
 * @param {NeuralAppearanceNodeMaterial} material - The neural appearance material.
 * @return {{ viewNormal: Node<vec3>, viewReflect: Node<vec3>, viewIrradiance: Node<vec3>, roughness: Node<float> }} Debug values.
 */
function evaluateNeuralDebugShading( material ) {

	const fragment = createNeuralFragmentContext( material );
	const canonicalNormal = evaluateLearnedCanonicalNormal( material, fragment );
	const query = material.neuralAppearanceData.outputs.ibl ?
		evaluateLearnedIBLQuery( material, fragment ) :
		{ direction: fragment.viewDirection.negate().reflect( canonicalNormal ).normalize(), roughness: TSL.float( 1 ) };

	return {
		viewNormal: canonicalToViewDirection( canonicalNormal ),
		viewReflect: canonicalToViewDirection( query.direction ),
		viewIrradiance: canonicalToViewDirection( TSL.vec3( 0, 0, 1 ) ),
		roughness: query.roughness
	};

}

function packDebugDirection( direction ) {

	return TSL.colorSpaceToWorking( TSL.vec4( TSL.packNormalToRGB( direction ), 1 ), THREE.SRGBColorSpace ).xyz;

}

function packDebugScalar( value ) {

	return TSL.colorSpaceToWorking( TSL.vec4( TSL.vec3( value ), 1 ), THREE.SRGBColorSpace ).xyz;

}

/**
 * Bilinear-samples every latent grid level texture at `uvNode` (ordinary
 * hardware bilinear + repeat wrap, no LOD/mip selection) - mirrors
 * `evaluateNeuralTextureRaw` in neural-texture/NeuralTextureNodeMaterial.js.
 * Returns one vec4 texel node per grid level, in level order (however many
 * levels this particular model has - see levelNames above); flatten with
 * `latentsFromTexels` to get the concatenated scalar latent array.
 */
function fetchLatentTexels( material, uvNode ) {

	const data = material.neuralAppearanceData;

	return data.latentTextures.map( ( levelTexture ) => TSL.texture( levelTexture, uvNode ).toVar() );

}

function fetchLatentCode( material ) {

	return latentsFromTexels( fetchLatentTexels( material, TSL.uv() ) );

}

// Built once per fragment (see NeuralAppearanceNodeMaterial's constructor and
// NeuralAppearanceLightingModel.start(), both of which now share a single
// instance of this context instead of each building their own) and reused by
// every decoder head -- BRDF, emission, opacity, IBL. `latents`/`frames` are
// included here so evaluateNeuralEmission/evaluateNeuralOpacity/evaluateNeuralIBL
// don't each independently re-sample the grid textures or re-run the
// rotation decoder (`buildDecoderFrames`) that BRDF also needs; only the
// directional BRDF Fn (which must run once per light, not once per fragment)
// still derives its own frames internally -- see createEvaluateNeuralBRDFFn.
function createNeuralFragmentContext( material ) {

	const uvNode = TSL.uv();
	const viewDirection = transformToCanonicalFrame( TSL.positionViewDirection ).toVar();
	const texels = fetchLatentTexels( material, uvNode );
	const latents = latentsFromTexels( texels );
	const frames = buildDecoderFrames( material.neuralAppearanceData.outputs.brdf, material._outputUniforms.brdf, latents );

	return {
		uvNode,
		viewDirection,
		texels,
		latents,
		frames
	};

}

function createOutputUniforms( outputs ) {

	const uniforms = {
		brdf: createHeadUniforms( outputs.brdf ),
		ibl: createHeadUniforms( outputs.ibl )
	};

	if ( outputs.indirectRadiance ) uniforms.indirectRadiance = createHeadUniforms( outputs.indirectRadiance );
	if ( outputs.indirectIrradiance ) uniforms.indirectIrradiance = createHeadUniforms( outputs.indirectIrradiance );
	if ( outputs.emission ) uniforms.emission = createHeadUniforms( outputs.emission );
	if ( outputs.opacity ) uniforms.opacity = createHeadUniforms( outputs.opacity );

	return uniforms;

}

function createHeadUniforms( decoder ) {

	const packed = packHeadParameters( decoder );

	// Two parallel uniform buffers rather than one combined buffer: a
	// `UniformArrayNode` is typed by a single `elementType`, so `mat4`
	// weight blocks and `vec4` bias vectors can't share one array the way
	// the old all-vec4 layout let them.
	return {
		weights: TSL.uniformArray( packed.weightsValues, 'mat4' ),
		biases: TSL.uniformArray( packed.biasesValues, 'vec4' ),
		rotationWeightsOffset: packed.rotationWeightsOffset,
		layers: packed.layers
	};

}

function isCompatibleNeuralAppearanceData( current, next ) {

	if ( ! current || ! next ) return false;
	if ( current.levels !== next.levels ) return false;
	if ( current.wrap !== next.wrap ) return false;
	if ( ! sameLatentTextureLayout( current.latentTextures, next.latentTextures ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.brdf, next.outputs.brdf ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.ibl, next.outputs.ibl ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.indirectRadiance, next.outputs.indirectRadiance ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.indirectIrradiance, next.outputs.indirectIrradiance ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.emission, next.outputs.emission ) ) return false;
	if ( ! sameHeadArchitecture( current.outputs.opacity, next.outputs.opacity ) ) return false;

	return true;

}

function sameLatentTextureLayout( currentTextures, nextTextures ) {

	if ( ! currentTextures || ! nextTextures || currentTextures.length !== nextTextures.length ) return false;

	for ( let i = 0; i < currentTextures.length; i ++ ) {

		if ( currentTextures[ i ].image.width !== nextTextures[ i ].image.width ) return false;
		if ( currentTextures[ i ].image.height !== nextTextures[ i ].image.height ) return false;
		if ( currentTextures[ i ].image.data.length !== nextTextures[ i ].image.data.length ) return false;

	}

	return true;

}

function sameHeadArchitecture( current, next ) {

	if ( Boolean( current ) !== Boolean( next ) ) return false;
	if ( ! current ) return true;
	if ( current.inputSize !== next.inputSize ) return false;
	if ( Boolean( current.rotation ) !== Boolean( next.rotation ) ) return false;

	if ( current.rotation ) {

		if ( current.rotation.inputSize !== next.rotation.inputSize ) return false;
		if ( current.rotation.outputSize !== next.rotation.outputSize ) return false;

	}

	if ( current.layers.length !== next.layers.length ) return false;

	for ( let i = 0; i < current.layers.length; i ++ ) {

		if ( current.layers[ i ].inputSize !== next.layers[ i ].inputSize ) return false;
		if ( current.layers[ i ].outputSize !== next.layers[ i ].outputSize ) return false;
		if ( current.layers[ i ].activation !== next.layers[ i ].activation ) return false;

	}

	const currentActivation = current.outputActivation || { type: 'linear' };
	const nextActivation = next.outputActivation || { type: 'linear' };

	if ( currentActivation.type !== nextActivation.type ) return false;
	if ( ( currentActivation.offset || 0 ) !== ( nextActivation.offset || 0 ) ) return false;
	if ( ( currentActivation.scale !== undefined ? currentActivation.scale : 1 ) !== ( nextActivation.scale !== undefined ? nextActivation.scale : 1 ) ) return false;
	if ( ( current.mode || 'mask' ) !== ( next.mode || 'mask' ) ) return false;
	if ( ( current.alphaCutoff !== undefined ? current.alphaCutoff : null ) !== ( next.alphaCutoff !== undefined ? next.alphaCutoff : null ) ) return false;

	return true;

}

function updateOutputUniforms( uniforms, outputs ) {

	updateHeadUniforms( uniforms.brdf, outputs.brdf );
	updateHeadUniforms( uniforms.ibl, outputs.ibl );
	if ( outputs.indirectRadiance ) updateHeadUniforms( uniforms.indirectRadiance, outputs.indirectRadiance );
	if ( outputs.indirectIrradiance ) updateHeadUniforms( uniforms.indirectIrradiance, outputs.indirectIrradiance );
	if ( outputs.emission ) updateHeadUniforms( uniforms.emission, outputs.emission );
	if ( outputs.opacity ) updateHeadUniforms( uniforms.opacity, outputs.opacity );

}

function updateHeadUniforms( uniforms, decoder ) {

	const packed = packHeadParameters( decoder );

	copyPackedVectors( uniforms.weights.array, packed.weightsValues );
	copyPackedVectors( uniforms.biases.array, packed.biasesValues );

}

function copyPackedVectors( targetArray, packed ) {

	if ( targetArray.length !== packed.length ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Packed uniform length mismatch (${ targetArray.length } !== ${ packed.length }).` );

	}

	for ( let i = 0; i < packed.length; i ++ ) {

		targetArray[ i ].copy( packed[ i ] );

	}

}

function copyLatentTextureData( destinationTextures, sourceTextures ) {

	for ( let i = 0; i < destinationTextures.length; i ++ ) {

		const destination = destinationTextures[ i ];
		const source = sourceTextures[ i ];

		destination.image.data.set( source.image.data );
		destination.needsUpdate = true;

	}

}

function packHeadParameters( decoder ) {

	const weightsValues = [];
	const biasesValues = [];
	const layers = [];
	let rotationWeightsOffset = null;

	if ( decoder.rotation ) {

		rotationWeightsOffset = weightsValues.length;
		weightsValues.push( ...packLayerWeightsMat4( decoder.rotation.weights, decoder.rotation.inputSize, decoder.rotation.outputSize ) );

	}

	for ( const layer of decoder.layers ) {

		const weightsOffset = weightsValues.length;
		weightsValues.push( ...packLayerWeightsMat4( layer.weights, layer.inputSize, layer.outputSize ) );

		const biasesOffset = biasesValues.length;
		biasesValues.push( ...packLayerBiasesVec4( layer.biases ) );

		layers.push( { weightsOffset, biasesOffset } );

	}

	return { weightsValues, biasesValues, rotationWeightsOffset, layers };

}

function buildDecoderInput( decoder, decoderUniforms, latents, wi, wo ) {

	const frames = buildDecoderFrames( decoder, decoderUniforms, latents );
	return projectDecoderInput( latents, frames, wi, wo, decoder.inputSize );

}

function buildDecoderFrames( decoder, decoderUniforms, latents ) {

	if ( decoder.rotation === null ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: A two-frame rotation decoder is required.' );

	}

	const rotation = unpackVec4Outputs(
		linearLayerPacked( packVec4Inputs( latents ), decoderUniforms, decoderUniforms.rotationWeightsOffset, null, latents.length, decoder.rotation.outputSize, 'linear' ),
		decoder.rotation.outputSize
	);
	const frames = [];

	for ( let frame = 0; frame < 2; frame ++ ) {

		const offset = frame * 6;
		const n = TSL.vec3( rotation[ offset ], rotation[ offset + 1 ], rotation[ offset + 2 ].add( 1 ) ).normalize().toVar();
		const t = TSL.vec3( rotation[ offset + 3 ].add( 1 ), rotation[ offset + 4 ], rotation[ offset + 5 ] ).normalize().toVar();
		const b = TSL.cross( n, t ).normalize().toVar();

		frames.push( { n, t, b } );

	}

	return frames;

}

function projectDecoderInput( latents, frames, wi, wo, inputSize ) {

	const input = latents.slice();

	for ( let frame = 0; frame < frames.length; frame ++ ) {

		const basis = frames[ frame ];

		input.push( wi.dot( basis.t ), wi.dot( basis.b ), wi.dot( basis.n ) );
		input.push( wo.dot( basis.t ), wo.dot( basis.b ), wo.dot( basis.n ) );

	}

	if ( input.length !== inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Decoder input has ${ input.length } values, expected ${ inputSize }.` );

	}

	return input;

}

function projectIBLInput( latents, frames, wo, inputSize ) {

	const input = latents.slice();

	for ( let frame = 0; frame < frames.length; frame ++ ) {

		const basis = frames[ frame ];
		input.push( wo.dot( basis.t ), wo.dot( basis.b ), wo.dot( basis.n ) );

	}

	if ( input.length !== inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: IBL decoder input has ${ input.length } values, expected ${ inputSize }.` );

	}

	return input;

}

function projectIndirectProbeInput( latents, wo, probe, inputSize ) {

	const input = latents.slice();

	input.push( wo.x, wo.y, wo.z );
	input.push( probe.x, probe.y, probe.z );

	if ( input.length !== inputSize ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: Indirect decoder input has ${ input.length } values, expected ${ inputSize }.` );

	}

	return input;

}

function evaluateIndirectProbeHead( material, head, uniforms, latents, wo, probe, label ) {

	const input = projectIndirectProbeInput( latents, wo, probe, head.inputSize );
	const decoded = toVec3( evaluateMLPViaFn( `evaluateIndirect${ label }Head_${ material.id }`, head.layers, uniforms, input, 3 ) );

	return applyOutputActivation( decoded, head.outputActivation );

}

function evaluateMLP( layers, uniforms, inputs ) {

	let activations = packVec4Inputs( inputs );

	for ( let i = 0; i < layers.length; i ++ ) {

		const layer = layers[ i ];
		const layerUniform = uniforms.layers[ i ];

		activations = linearLayerPacked( activations, uniforms, layerUniform.weightsOffset, layerUniform.biasesOffset, layer.inputSize, layer.outputSize, layer.activation );

	}

	return unpackVec4Outputs( activations, layers[ layers.length - 1 ].outputSize );

}

// `linearLayerPacked` materializes every layer's output with `.toVar()` (see
// its own comment) so a deep net's per-layer expressions don't compound into
// one huge inline tree - but that only actually *frees* those locals if the
// evaluation happens inside its own `TSL.Fn()` scope, the way
// createEvaluateNeuralBRDFFn already wraps the BRDF decoder. Call sites that
// instead call `evaluateMLP` directly inline (no Fn wrapper) accumulate all
// of that net's materialized locals into the *caller's* scope - the main
// fragment shader body, for evaluateNeuralIBLForTexels - and when several
// such nets run side by side there (iblHead + indirectRadiance +
// indirectIrradiance), their locals all pile up in that one shared scope at
// once and can blow WGSL's private-address-space budget (observed: "The
// combined byte size of all variables in the private address space exceeds
// 8192 bytes", pipeline creation failure, nothing renders).
//
// This wraps a single MLP head's evaluation in its own `TSL.Fn()`. so its
// `.toVar()` locals live and die inside that head's own function scope
// (compiled once as a real WGSL function, called once here) instead of
// piling up in the caller's scope alongside every other head's locals. Only
// suitable for outputSize in {1, 3, 4} (float/vec3/vec4 - what a TSL.Fn can
// return as a single value); every current MLP head fits one of those.
function evaluateMLPViaFn( name, layers, uniforms, inputs, outputSize ) {

	const packedInputs = packVec4Inputs( inputs );
	const inputNames = packedInputs.map( ( _, i ) => `in${ i }` );

	const outputType = outputSize === 1 ? 'float' : outputSize === 3 ? 'vec3' :
		outputSize === 4 ? 'vec4' : null;

	if ( outputType === null ) {

		throw new Error( `THREE.NeuralAppearanceNodeMaterial: evaluateMLPViaFn only supports outputSize 1/3/4 (got ${ outputSize }).` );

	}

	const fn = TSL.Fn( ( args ) => {

		let activations = inputNames.map( n => args[ n ] );

		for ( let i = 0; i < layers.length; i ++ ) {

			const layer = layers[ i ];
			const layerUniform = uniforms.layers[ i ];

			activations = linearLayerPacked( activations, uniforms, layerUniform.weightsOffset, layerUniform.biasesOffset, layer.inputSize, layer.outputSize, layer.activation );

		}

		const unpacked = unpackVec4Outputs( activations, outputSize );

		if ( outputType === 'float' ) return unpacked[ 0 ];
		if ( outputType === 'vec3' ) return TSL.vec3( unpacked[ 0 ], unpacked[ 1 ], unpacked[ 2 ] );
		return TSL.vec4( unpacked[ 0 ], unpacked[ 1 ], unpacked[ 2 ], unpacked[ 3 ] );

	} ).setLayout( {
		name,
		type: outputType,
		inputs: inputNames.map( n => ( { name: n, type: 'vec4' } ) )
	} );

	const result = fn( ...packedInputs );

	if ( outputType === 'float' ) return [ result ];
	if ( outputType === 'vec3' ) return [ result.x, result.y, result.z ];
	return [ result.x, result.y, result.z, result.w ];

}

// This module's per-head uniform layout (`uniforms.weights`/`uniforms.biases`
// + `weightsOffset`/`biasesOffset`, see packHeadParameters/createHeadUniforms)
// is specific to neural-appearance - all of a head's layers (plus its
// rotation layer, if any) share one mat4 weights buffer and one vec4 biases
// buffer, addressed by per-layer offset, rather than each layer getting its
// own uniformArray the way neural-texture's simpler single-network case does
// - so this stays a local wrapper around the shared evaluateLinearLayerMat4
// rather than a bare re-export: it supplies the `getWeightMat4`/`getBiasVec4`
// closures that read from those two buffers. `biasesOffset: null` (only the
// rotation layer - see buildDecoderFrames - which packs no bias values at
// all, matching the training kernel's bias-free rotation projection) skips
// the bias term entirely rather than reading a bogus offset.
function linearLayerPacked( inputs, uniforms, weightsOffset, biasesOffset, inputSize, outputSize, activation ) {

	const inputVectorCount = Math.ceil( inputSize / 4 );

	return evaluateLinearLayerMat4(
		inputs, inputSize, outputSize, activation,
		( outputVector, inputVector ) => uniforms.weights.element( weightsOffset + outputVector * inputVectorCount + inputVector ),
		biasesOffset !== null ? ( outputVector ) => uniforms.biases.element( biasesOffset + outputVector ) : null
	);

}

function toVec3( values ) {

	if ( values.length !== 3 ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: Decoder output must be RGB.' );

	}

	return TSL.vec3( values[ 0 ], values[ 1 ], values[ 2 ] );

}

function applyOutputActivation( value, activation ) {

	if ( activation.type === 'exp' ) {

		return TSL.exp( value.add( activation.offset || 0 ) );

	}

	if ( activation.type === 'scaledSigmoid' ) {

		const scale = activation.scale !== undefined ? activation.scale : 1;
		return sigmoidTSL( value ).mul( scale );

	}

	return value.max( 0 );

}

function applyScalarOutputActivation( value, activation ) {

	if ( activation.type === 'sigmoid' ) {

		return sigmoidTSL( value );

	}

	return applyOutputActivation( value, activation );

}

export {
	evaluateNeuralBRDF,
	evaluateNeuralEmission,
	evaluateNeuralIBL,
	evaluateNeuralOpacity,
	evaluateNeuralDebugShading,
	packDebugDirection,
	packDebugScalar,
	createEvaluateNeuralBRDFFn,
	createNeuralFragmentContext,
	transformToCanonicalFrame,
	fetchLatentCode,
	createOutputUniforms,
	createHeadUniforms,
	isCompatibleNeuralAppearanceData,
	updateOutputUniforms,
	copyLatentTextureData,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	buildDecoderInput,
	projectIBLInput,
	canonicalToViewDirection,
	canonicalToWorldDirection,
	evaluateMLP,
	packVec4Inputs,
	unpackVec4Outputs,
	linearLayerPacked,
	toVec3,
	applyOutputActivation,
	applyScalarOutputActivation
};

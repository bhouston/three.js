import * as THREE from 'three';
import * as TSL from 'three/tsl';
import { LEVELS } from './NeuralAppearanceFormat.js';

// The multiresolution latent grid always has LEVELS (=4) levels of
// CHANNELS_PER_LEVEL (=4) features each - same fixed geometry as
// neural-texture / neural-material (see NeuralGridModel.js) - so the runtime
// evaluator below is written directly against 4 named vec4 texel inputs
// (latent0..latent3) rather than looping over a variable-length array; this
// keeps the TSL graph shape identical to the shared MLP evaluator helpers.
const LEVEL_NAMES = [ 'latent0', 'latent1', 'latent2', 'latent3' ];

function createEvaluateNeuralBRDFFn( material ) {

	const brdf = material.neuralAppearanceData.outputs.brdf;
	const uniforms = material._outputUniforms.brdf;

	return TSL.Fn( ( { wi, wo, latent0, latent1, latent2, latent3 } ) => {

		const latents = latentsFromTexels( latent0, latent1, latent2, latent3 );
		const frames = buildDecoderFrames( brdf, uniforms, latents );
		const input = projectDecoderInput( latents, frames, wi, wo, brdf.inputSize );
		const decoded = toVec3( evaluateMLP( brdf.layers, uniforms, input ) );

		return applyOutputActivation( decoded, brdf.outputActivation ).mul( wi.z.max( 0 ) );

	} ).setLayout( {
		name: `evaluateNeuralBRDF_${ material.id }`,
		type: 'vec3',
		inputs: [
			{ name: 'wi', type: 'vec3' },
			{ name: 'wo', type: 'vec3' },
			{ name: 'latent0', type: 'vec4' },
			{ name: 'latent1', type: 'vec4' },
			{ name: 'latent2', type: 'vec4' },
			{ name: 'latent3', type: 'vec4' }
		]
	} );

}

function evaluateNeuralBRDF( material, lightDirection, context, evaluateFn ) {

	const fragment = context || createNeuralFragmentContext( material );
	const fn = evaluateFn || createEvaluateNeuralBRDFFn( material );
	const wi = transformToCanonicalFrame( lightDirection );

	return fn( wi, fragment.viewDirection, fragment.texel0, fragment.texel1, fragment.texel2, fragment.texel3 );

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

// `latents`/`frames` are the same per-fragment values createNeuralFragmentContext
// already computed for BRDF -- passed in directly instead of re-sampling the
// grid textures and re-running the rotation decoder (buildDecoderFrames)
// here a second time. Not exported/called outside this module, so its
// signature is free to change.
//
// PERF PROFILING: `material.neuralPerfDebug` (see webgpu_materials_neural_appearance.html's
// "perf (IBL stages)" GUI folder) can force any of this function's 5 GPU-cost
// stages -- the iblHead MLP query, the two envNode.context().isolate() PMREM
// samples, and the indirectRadiance/indirectIrradiance MLP heads -- to be
// skipped and replaced with a cheap constant, so each stage's frame-time
// contribution can be measured in isolation. Purely a debugging aid: when
// `neuralPerfDebug` is unset (the default), every flag below is falsy and
// this function's shader graph and output are byte-for-byte what they were
// before this instrumentation was added.
function evaluateNeuralIBLForTexels( material, envNode, wo, latents, frames, isolate = 'full' ) {

	const perf = material.neuralPerfDebug || {};

	const ibl = material.neuralAppearanceData.outputs.ibl;
	const uniforms = material._outputUniforms.ibl;

	let queryDirection, queryRoughness;

	if ( perf.skipIblHead ) {

		queryDirection = TSL.vec3( 0, 0, 1 );
		queryRoughness = TSL.float( 0.5 );

	} else {

		const queryInput = projectIBLInput( latents, frames, wo, ibl.inputSize );
		const query = evaluateMLPViaFn( `evaluateIBLHead_${ material.id }`, ibl.layers, uniforms, queryInput, 4 );
		queryDirection = TSL.vec3( query[ 0 ], query[ 1 ], query[ 2 ] ).normalize();
		queryRoughness = TSL.float( 1 ).div( TSL.float( 1 ).add( TSL.exp( query[ 3 ].negate() ) ) );

	}

	const incoming = perf.skipIncomingSample ? TSL.vec3( 0.5 ) : envNode.context( {
		getUV: () => canonicalToWorldDirection( queryDirection ),
		getTextureLevel: () => queryRoughness
	} ).isolate().mul( TSL.materialEnvIntensity );
	const irradiance = perf.skipIrradianceSample ? TSL.vec3( 0.5 ) : envNode.context( {
		getUV: () => canonicalToWorldDirection( TSL.vec3( 0, 0, 1 ) ),
		getTextureLevel: () => TSL.float( 1 )
	} ).isolate().mul( TSL.materialEnvIntensity );
	const radianceHead = material.neuralAppearanceData.outputs.indirectRadiance;
	const irradianceHead = material.neuralAppearanceData.outputs.indirectIrradiance;
	let outgoing = TSL.vec3( 0 );

	if ( isolate !== 'irradiance' && radianceHead ) {

		outgoing = outgoing.add( perf.skipRadianceHead ? incoming :
			evaluateIndirectProbeHead( material, radianceHead, material._outputUniforms.indirectRadiance, latents, wo, incoming, 'Radiance' ) );

	}

	if ( isolate !== 'radiance' && irradianceHead ) {

		outgoing = outgoing.add( perf.skipIrradianceHead ? irradiance :
			evaluateIndirectProbeHead( material, irradianceHead, material._outputUniforms.indirectIrradiance, latents, wo, irradiance, 'Irradiance' ) );

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

function latentsFromTexels( texel0, texel1, texel2, texel3 ) {

	return [
		texel0.x, texel0.y, texel0.z, texel0.w,
		texel1.x, texel1.y, texel1.z, texel1.w,
		texel2.x, texel2.y, texel2.z, texel2.w,
		texel3.x, texel3.y, texel3.z, texel3.w
	];

}

function evaluateLearnedCanonicalNormal( material, fragment ) {

	const brdf = material.neuralAppearanceData.outputs.brdf;
	const uniforms = material._outputUniforms.brdf;

	return buildDecoderFrames( brdf, uniforms, latentsFromTexels( fragment.texel0, fragment.texel1, fragment.texel2, fragment.texel3 ) )[ 0 ].n;

}

function sigmoidNode( value ) {

	return TSL.float( 1 ).div( TSL.float( 1 ).add( TSL.exp( value.negate() ) ) );

}

function evaluateLearnedIBLQueryForTexels( material, texel0, texel1, texel2, texel3, wo ) {

	const latents = latentsFromTexels( texel0, texel1, texel2, texel3 );
	const ibl = material.neuralAppearanceData.outputs.ibl;
	const frames = buildDecoderFrames( material.neuralAppearanceData.outputs.brdf, material._outputUniforms.brdf, latents );
	const input = projectIBLInput( latents, frames, wo, ibl.inputSize );
	const output = evaluateMLPViaFn( `evaluateIBLHead_${ material.id }`, ibl.layers, material._outputUniforms.ibl, input, 4 );

	return {
		direction: TSL.vec3( output[ 0 ], output[ 1 ], output[ 2 ] ).normalize(),
		roughness: sigmoidNode( output[ 3 ] )
	};

}

function evaluateLearnedIBLQuery( material, fragment ) {

	return evaluateLearnedIBLQueryForTexels( material, fragment.texel0, fragment.texel1, fragment.texel2, fragment.texel3, fragment.viewDirection );

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
 * hardware bilinear + repeat wrap, no LOD/mip selection) and concatenates
 * their channels into a flat array of scalar latent nodes - mirrors
 * `evaluateNeuralTextureRaw` in neural-texture/NeuralTextureNodeMaterial.js.
 */
function fetchLatentTexels( material, uvNode ) {

	const data = material.neuralAppearanceData;
	const texels = data.latentTextures.map( ( levelTexture ) => TSL.texture( levelTexture, uvNode ).toVar() );

	return { texel0: texels[ 0 ], texel1: texels[ 1 ], texel2: texels[ 2 ], texel3: texels[ 3 ] };

}

function fetchLatentCode( material ) {

	const texels = fetchLatentTexels( material, TSL.uv() );

	return latentsFromTexels( texels.texel0, texels.texel1, texels.texel2, texels.texel3 );

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
	const latents = latentsFromTexels( texels.texel0, texels.texel1, texels.texel2, texels.texel3 );
	const frames = buildDecoderFrames( material.neuralAppearanceData.outputs.brdf, material._outputUniforms.brdf, latents );

	return {
		viewDirection,
		texel0: texels.texel0,
		texel1: texels.texel1,
		texel2: texels.texel2,
		texel3: texels.texel3,
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

	return {
		parameters: TSL.uniformArray( packed.values, 'vec4' ),
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

	copyPackedVectors( uniforms.parameters.array, packHeadParameters( decoder ).values );

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

function packLayerWeights( weights, inputSize, outputSize ) {

	const inputVectorCount = Math.ceil( inputSize / 4 );
	const packed = [];

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			const inputBase = vectorIndex * 4;
			const rowOffset = outputIndex * inputSize;

			packed.push( new THREE.Vector4(
				inputBase < inputSize ? ( weights[ rowOffset + inputBase ] || 0 ) : 0,
				inputBase + 1 < inputSize ? ( weights[ rowOffset + inputBase + 1 ] || 0 ) : 0,
				inputBase + 2 < inputSize ? ( weights[ rowOffset + inputBase + 2 ] || 0 ) : 0,
				inputBase + 3 < inputSize ? ( weights[ rowOffset + inputBase + 3 ] || 0 ) : 0
			) );

		}

	}

	return packed;

}

function packLayerBiases( biases ) {

	const packed = [];
	const vectorCount = Math.ceil( biases.length / 4 );

	for ( let vectorIndex = 0; vectorIndex < vectorCount; vectorIndex ++ ) {

		const offset = vectorIndex * 4;

		packed.push( new THREE.Vector4(
			biases[ offset ] || 0,
			biases[ offset + 1 ] || 0,
			biases[ offset + 2 ] || 0,
			biases[ offset + 3 ] || 0
		) );

	}

	return packed;

}

function packHeadParameters( decoder ) {

	const values = [];
	const layers = [];
	let rotationWeightsOffset = null;

	if ( decoder.rotation ) {

		rotationWeightsOffset = values.length;
		values.push( ...packLayerWeights( decoder.rotation.weights, decoder.rotation.inputSize, decoder.rotation.outputSize ) );

	}

	for ( const layer of decoder.layers ) {

		const weightsOffset = values.length;
		values.push( ...packLayerWeights( layer.weights, layer.inputSize, layer.outputSize ) );

		const biasesOffset = values.length;
		values.push( ...packLayerBiases( layer.biases ) );

		layers.push( { weightsOffset, biasesOffset } );

	}

	return { values, rotationWeightsOffset, layers };

}

function buildDecoderInput( decoder, decoderUniforms, latents, wi, wo ) {

	const frames = buildDecoderFrames( decoder, decoderUniforms, latents );
	return projectDecoderInput( latents, frames, wi, wo, decoder.inputSize );

}

function buildDecoderFrames( decoder, decoderUniforms, latents ) {

	if ( decoder.rotation === null ) {

		throw new Error( 'THREE.NeuralAppearanceNodeMaterial: A two-frame rotation decoder is required.' );

	}

	const rotation = unpackNodeInputs(
		linearLayerPacked( packNodeInputs( latents ), decoderUniforms.parameters, decoderUniforms.rotationWeightsOffset, null, latents.length, decoder.rotation.outputSize, 'linear' ),
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

	let activations = packNodeInputs( inputs );

	for ( let i = 0; i < layers.length; i ++ ) {

		const layer = layers[ i ];
		const layerUniform = uniforms.layers[ i ];

		activations = linearLayerPacked( activations, uniforms.parameters, layerUniform.weightsOffset, layerUniform.biasesOffset, layer.inputSize, layer.outputSize, layer.activation );

	}

	return unpackNodeInputs( activations, layers[ layers.length - 1 ].outputSize );

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

	const packedInputs = packNodeInputs( inputs );
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

			activations = linearLayerPacked( activations, uniforms.parameters, layerUniform.weightsOffset, layerUniform.biasesOffset, layer.inputSize, layer.outputSize, layer.activation );

		}

		const unpacked = unpackNodeInputs( activations, outputSize );

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

function packNodeInputs( inputs ) {

	const inputVectors = [];
	const inputVectorCount = Math.ceil( inputs.length / 4 );

	for ( let i = 0; i < inputVectorCount; i ++ ) {

		const offset = i * 4;

		inputVectors.push( TSL.vec4(
			inputs[ offset ] || 0,
			inputs[ offset + 1 ] || 0,
			inputs[ offset + 2 ] || 0,
			inputs[ offset + 3 ] || 0
		) );

	}

	return inputVectors;

}

function unpackNodeInputs( inputs, outputSize ) {

	const outputs = [];

	for ( let outputIndex = 0; outputIndex < outputSize; outputIndex ++ ) {

		const vector = inputs[ Math.floor( outputIndex / 4 ) ];
		outputs.push( vector.element( outputIndex % 4 ) );

	}

	return outputs;

}

function linearLayerPacked( inputs, parameters, weightsOffset, biasesOffset, inputSize, outputSize, activation ) {

	const outputs = [];
	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );

	for ( let outputVector = 0; outputVector < outputVectorCount; outputVector ++ ) {

		const outputBase = outputVector * 4;
		const sums = [ TSL.float( 0 ), TSL.float( 0 ), TSL.float( 0 ), TSL.float( 0 ) ];

		for ( let vectorIndex = 0; vectorIndex < inputVectorCount; vectorIndex ++ ) {

			const inputVector = inputs[ vectorIndex ];

			for ( let component = 0; component < 4; component ++ ) {

				const outputIndex = outputBase + component;

				if ( outputIndex < outputSize ) {

					sums[ component ] = sums[ component ].add( TSL.dot( inputVector, parameters.element( weightsOffset + outputIndex * inputVectorCount + vectorIndex ) ) );

				}

			}

		}

		// `biasesOffset: null` (only the rotation layer - see
		// buildDecoderFrames - which packs no bias values at all, matching
		// the training kernel's bias-free rotation projection) skips the
		// bias term entirely rather than reading a bogus offset.
		const bias = biasesOffset !== null ? parameters.element( biasesOffset + outputVector ) : TSL.vec4( 0 );
		let value = bias.add( TSL.vec4( sums[ 0 ], sums[ 1 ], sums[ 2 ], sums[ 3 ] ) );

		if ( activation === 'relu' ) {

			value = value.max( 0 );

		}

		// Materialize each output vector4 into a variable before it's consumed by
		// the next layer's dot() calls, instead of leaving it as a live expression
		// that gets re-expanded at every downstream reference. Mirrors the fix
		// neural-material's evaluateNeuralTextureRaw (NeuralTextureNodeMaterial.js)
		// applies for the same reason: without it, per-layer expressions compound
		// across layers into a much larger (and, for a deep enough net, WGSL
		// parser-recursion-breaking) generated shader.
		outputs.push( value.toVar() );

	}

	return outputs;

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
		return TSL.float( scale ).div( TSL.float( 1 ).add( TSL.exp( value.negate() ) ) );

	}

	return value.max( 0 );

}

function applyScalarOutputActivation( value, activation ) {

	if ( activation.type === 'sigmoid' ) {

		return TSL.float( 1 ).div( TSL.float( 1 ).add( TSL.exp( value.negate() ) ) );

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
	packLayerWeights,
	packLayerBiases,
	buildDecoderInput,
	projectIBLInput,
	canonicalToViewDirection,
	canonicalToWorldDirection,
	evaluateMLP,
	packNodeInputs,
	unpackNodeInputs,
	linearLayerPacked,
	toVec3,
	applyOutputActivation,
	applyScalarOutputActivation,
	LEVEL_NAMES,
	LEVELS
};
